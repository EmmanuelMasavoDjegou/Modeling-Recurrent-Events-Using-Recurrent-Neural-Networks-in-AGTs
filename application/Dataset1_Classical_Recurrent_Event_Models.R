###############################################################################
# Dataset 1 (colorectal cancer readmission, frailtypack::readmission)
# Classical recurrent-event models, split-aware.
#
# DESIGN
# ------
# `summary(fit)$concordance` reports Harrell's C computed IN SAMPLE, unweighted,
# with no train/test split. RNN-AGT's reported C-index is OUT OF SAMPLE and
# IPCW-weighted. Those are different estimands and must not be compared: an
# in-sample number is optimistic by construction, most severely on small
# samples.
#
# This script therefore:
#
#   1. Reads split assignments written by the Python driver, so the Cox
#      models are fitted on exactly the same training partitions as AGT-WRS,
#      NN-AGT and RNN-AGT. Table 7's paired differences are only computable
#      if every method sees the same splits.
#
#   2. Does NOT compute a concordance index. It exports the linear
#      predictor on the held-out rows and lets Python compute the IPCW C-index
#      with the identical estimator used for the neural models. Computing it
#      separately here would reintroduce exactly the estimand mismatch above.
#
#   3. Keeps the full-data fits as a separate, clearly labelled block: the
#      coefficient estimates and standard errors are worth reporting for
#      interpretation, but are not the basis of any predictive comparison.
#
# SIGN CONVENTION (important)
# ---------------------------
# A Cox linear predictor is on the log-hazard scale: LARGER means higher
# hazard, hence SHORTER gap times. The AFT models predict log gap time:
# LARGER means LONGER. Python negates the exported linear predictor before
# computing concordance. Do not negate it here as well.
#
# USAGE
# -----
#   Rscript Dataset1_Classical_Recurrent_Event_Models.R \
#       --data data/crc.csv --splits results/splits/splits_crc.csv --out results/cox_lp_crc.csv
#
# If --splits is omitted, only the full-data descriptive block runs.
###############################################################################

###############################################################################
# PACKAGE BOOTSTRAP
#
# Installs anything missing before loading it, so the script runs on a clean
# machine without a separate setup step. Only packages not already present are
# fetched, so re-runs cost nothing.
#
# If a package fails to build, the message below points at the usual cause:
# a missing system library, not a problem with R. On Debian/Ubuntu the ones
# this dependency tree needs are
#
#   sudo apt install -y libuv1-dev libcurl4-openssl-dev libssl-dev \
#                       libxml2-dev libfontconfig1-dev libfreetype6-dev \
#                       libharfbuzz-dev libfribidi-dev libpng-dev \
#                       libtiff-dev libjpeg-dev
#
# libuv1-dev in particular is required by `fs`, which `frailtypack` pulls in
# via shiny; without it the install fails with "uv.h: No such file or
# directory" and takes sass, bslib, shiny and frailtypack down with it.
###############################################################################

REQUIRED <- c("survival", "frailtypack", "dplyr")

install_if_missing <- function(pkgs, repos = "https://cloud.r-project.org") {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0L) {
    message("All required packages already installed.")
    return(invisible(TRUE))
  }
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = repos)

  still <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still) > 0L) {
    stop(
      "Failed to install: ", paste(still, collapse = ", "), "\n",
      "install.packages() only warns on build failure, so scroll up to the\n",
      "first 'ERROR:' line for the real cause. It is usually a missing system\n",
      "library; see the header of this file for the apt packages needed.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

install_if_missing(REQUIRED)

suppressPackageStartupMessages({
  library(survival)
  library(frailtypack)
  library(dplyr)
})

# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA_character_) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}
data_path   <- get_arg("--data",   "data/crc.csv")
splits_path <- get_arg("--splits", NA_character_)
out_path    <- get_arg("--out",    "results/cox_lp_crc.csv")

# --------------------------------------------------------------------------
# Load and format
# --------------------------------------------------------------------------
if (file.exists(data_path)) {
  cat("Reading preprocessed data from", data_path, "\n")
  formatted_data <- read.csv(data_path)
  # data/crc.csv from the preprocessing script carries:
  #   id, time (= t.stop), gap_time, event, Z1, Z2, Z3, Z4, tau
  formatted_data <- formatted_data %>%
    rename(stop = time, gtime = gap_time, status = event) %>%
    arrange(id, stop) %>%
    group_by(id) %>%
    mutate(
      start = dplyr::lag(stop, default = 0),
      order = seq_along(stop)
    ) %>%
    ungroup()
} else {
  cat("No", data_path, "found; rebuilding from frailtypack::readmission\n")
  data(readmission, package = "frailtypack")
  readmission$chemo <- ifelse(readmission$chemo == "Treated", 1, 0)
  readmission$sex   <- ifelse(readmission$sex == "Female", 1, 0)
  readmission$dukes <- as.numeric(readmission$dukes)

  readmission <- readmission[order(readmission$id, readmission$t.stop), ]
  last_index <- !duplicated(readmission$id, fromLast = TRUE)
  readmission$event[last_index] <- 0

  formatted_data <- readmission %>%
    dplyr::select(id, t.start, t.stop, event, chemo, sex, dukes, time) %>%
    rename(start = t.start, stop = t.stop, gtime = time, status = event,
           Z1 = chemo, Z2 = sex, Z3 = dukes) %>%
    mutate(id = as.numeric(factor(id)),
           order = ave(id, id, FUN = seq_along))
}

# Guard: gap times must be strictly positive, since the AFT side works on the
# log scale. Dropping these silently would put the two model families on
# different row sets and quietly break the pairing.
n_bad <- sum(formatted_data$gtime <= 0, na.rm = TRUE)
if (n_bad > 0) {
  stop(sprintf(
    paste("%d rows have non-positive gap times. Resolve these in the",
          "preprocessing script rather than here, so that the R and Python",
          "sides operate on identical rows."), n_bad))
}

cat("Subjects:", length(unique(formatted_data$id)),
    "| records:", nrow(formatted_data), "\n")

COVARIATES <- c("Z1", "Z2", "Z3")

# --------------------------------------------------------------------------
# Block A: full-data fits, for coefficient interpretation only
# --------------------------------------------------------------------------
cat("\n=== Full-data fits (coefficients only; NOT used for prediction) ===\n")

fit_full <- list(
  AG     = coxph(Surv(start, stop, status) ~ Z1 + Z2 + Z3,
                 data = formatted_data),
  LWYY   = coxph(Surv(start, stop, status) ~ Z1 + Z2 + Z3 + cluster(id),
                 data = formatted_data),
  WLW    = coxph(Surv(stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                 data = formatted_data),
  PWP_TT = coxph(Surv(start, stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                 data = formatted_data),
  PWP_GT = coxph(Surv(gtime, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                 data = formatted_data)
)

for (nm in names(fit_full)) {
  cat("\n---", nm, "---\n")
  print(summary(fit_full[[nm]])$coefficients)
}

cat("\nNOTE: the in-sample concordance printed by summary() is deliberately\n")
cat("not reported here. It is not comparable with the out-of-sample IPCW\n")
cat("C-index used for the AFT models; see the header of this file.\n")

saveRDS(fit_full, file = "results/cox_full_fits_crc.rds")

# --------------------------------------------------------------------------
# Block B: split-aware fits, exporting held-out linear predictors
# --------------------------------------------------------------------------
if (is.na(splits_path) || !file.exists(splits_path)) {
  cat("\nNo --splits file supplied; skipping the split-aware block.\n")
  cat("Run the Python driver first to generate split assignments.\n")
  quit(status = 0)
}

cat("\n=== Split-aware fits from", splits_path, "===\n")
splits <- read.csv(splits_path)   # columns: split_id, id, partition
stopifnot(all(c("split_id", "id", "partition") %in% names(splits)))

fit_and_predict <- function(train_df, test_df) {
  # Returns a data frame of held-out linear predictors, one row per test
  # record per model. A model that fails to converge on a given split yields
  # NA rather than aborting the run; the Python side counts and reports them.
  specs <- list(
    WLW    = function(d) coxph(Surv(stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                               data = d),
    PWP_TT = function(d) coxph(Surv(start, stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                               data = d),
    PWP_GT = function(d) coxph(Surv(gtime, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
                               data = d)
  )

  out <- list()
  for (nm in names(specs)) {
    lp <- rep(NA_real_, nrow(test_df))
    fit <- try(specs[[nm]](train_df), silent = TRUE)
    if (!inherits(fit, "try-error")) {
      # Strata present in test but not in train give NA; that is correct
      # behaviour, not an error. It happens when a split leaves no subject
      # with, say, a 6th event in the training partition.
      pred <- try(predict(fit, newdata = test_df, type = "lp"), silent = TRUE)
      if (!inherits(pred, "try-error")) lp <- as.numeric(pred)
    }
    out[[nm]] <- data.frame(
      model = nm,
      id    = test_df$id,
      order = test_df$order,
      lp    = lp,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

split_ids <- sort(unique(splits$split_id))
results <- vector("list", length(split_ids))

for (k in seq_along(split_ids)) {
  b <- split_ids[k]
  assign_b <- splits[splits$split_id == b, ]
  train_ids <- assign_b$id[assign_b$partition == "train"]
  test_ids  <- assign_b$id[assign_b$partition == "test"]

  train_df <- formatted_data[formatted_data$id %in% train_ids, ]
  test_df  <- formatted_data[formatted_data$id %in% test_ids, ]

  # Strata must be levels seen in training, else predict() cannot evaluate.
  # Collapsing the tail of the event-order distribution is the standard remedy
  # and is applied identically to both partitions.
  max_order <- max(train_df$order)
  train_df$order <- pmin(train_df$order, max_order)
  test_df$order  <- pmin(test_df$order,  max_order)

  res <- fit_and_predict(train_df, test_df)
  res$split_id <- b
  results[[k]] <- res

  if (k %% 20 == 0 || k == length(split_ids)) {
    cat("  split", k, "/", length(split_ids), "\n")
  }
}

all_lp <- do.call(rbind, results)
write.csv(all_lp, out_path, row.names = FALSE)

n_na <- sum(is.na(all_lp$lp))
cat("\nWrote", nrow(all_lp), "rows to", out_path, "\n")
cat("Missing linear predictors:", n_na,
    sprintf("(%.2f%%)\n", 100 * n_na / nrow(all_lp)))
if (n_na > 0) {
  cat("These arise where a test record's event order was not represented in\n")
  cat("the training partition. Python excludes them and reports the count.\n")
}
