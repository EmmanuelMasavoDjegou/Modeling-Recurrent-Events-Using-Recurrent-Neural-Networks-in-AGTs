###############################################################################
# Dataset 2: chronic granulomatous disease (survival::cgd)
#
# CHANGES IN THIS REVISION
# ------------------------
# The exploratory and plotting sections are unchanged. Three additions at the
# end make the output directly consumable by the Python drivers:
#
#   1. A `delta` column is written alongside `event`. The Python loader expects
#      `delta` (1 = fully observed gap, 0 = censored); keeping both avoids a
#      silent mismatch if either side is edited later.
#
#   2. Non-positive gap times now raise rather than being dropped quietly. The
#      AFT models work on the log scale, so a zero gap is not representable.
#      Dropping it here while the Cox side keeps it would put the two model
#      families on different row sets and break the pairing that Table 8's
#      paired differences depend on.
#
#   3. A short integrity report is printed: subjects, records, events, and the
#      censoring fraction. These are the numbers quoted in Section 5.1, so
#      they should be checked against the manuscript rather than assumed.
###############################################################################

# Paths below are relative to the REPOSITORY ROOT. Run this script from
# there:  Rscript application/Dataset2_Data_Preprocessing.R
if (!dir.exists("data"))    dir.create("data", recursive = TRUE)
if (!dir.exists("results")) dir.create("results", recursive = TRUE)

# Load necessary libraries
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

# Only the packages this script actually calls are required.
REQUIRED <- c("frailtypack", "survival", "dplyr")

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

library(frailtypack)
library(survival)
library(dplyr)

# Load the CGD dataset
data(cgd, package = "survival")
head(cgd)
str(cgd)

############################################
# Step 1: Subset Data and Preprocessing
############################################

# Subset the data with relevant columns
cgd_data <- cgd %>%
  dplyr::select(id, tstart, tstop, status, enum, sex, age, treat) %>%
  rename(
    start = tstart,
    time = tstop,      # Rename tstop to time
    event = status,
    Z1 = treat,        # Use treatment as chemo variable
    Z2 = sex,          # Use sex as gender variable
    Z3 = age           # Use age as baseline numeric variable
  )

# Encode categorical variables:
# Z1 (treatment): Placebo = 0, rIFN-g = 1
# Z2 (sex): Male = 0, Female = 1
cgd_data <- cgd_data %>%
  mutate(
    Z1 = ifelse(Z1 == "placebo", 0, 1),
    Z2 = ifelse(Z2 == "male", 0, 1)
  )

# Add the final censored time (tau) for each patient
cgd_data <- cgd_data %>%
  group_by(id) %>%
  mutate(tau = max(time)) %>%
  ungroup()

# Check for rows with zero-length intervals and remove them
cgd_data <- cgd_data %>%
  filter(start != time)

# Assign sequential IDs starting from 1
cgd_data <- cgd_data %>%
  mutate(id = as.numeric(factor(id)))

############################################
# Step 1A: Calculate the Gap Time
############################################

# Calculate gap_time as the difference between the current and previous time
cgd_data <- cgd_data %>%
  group_by(id) %>%
  mutate(gap_time = ifelse(row_number() == 1, time, time - lag(time))) %>%
  ungroup()

# View the first few rows
cat("First few rows of the preprocessed CGD dataset:\n")
print(head(cgd_data))

############################################
# Step 2: Analyze Events Per Patient
############################################

# Count the number of events per patient
events_per_patient <- table(cgd_data$id)

# Summarize the number of events per patient
cat("Summary of events per patient:\n")
print(summary(as.numeric(events_per_patient)))

# Identify the patient(s) with the maximum number of events
max_events_patient <- names(which.max(events_per_patient))
max_events_count <- max(events_per_patient)

cat("Patient ID with the maximum number of events:", max_events_patient, "\n")
cat("Number of events for this patient:", max_events_count, "\n")

############################################
# Step 3: Visualizations
############################################

# Export histogram and boxplot as PNG
png("cgd_histograms.png", width = 800, height = 400)
par(mfrow = c(1, 2))
hist(as.numeric(events_per_patient), 
     breaks = 30, 
     main = "Events per patient (CGD dataset)", 
     xlab = "Number of Events", 
     col = "steelblue")
boxplot(as.numeric(events_per_patient), 
        main = "Boxplot - Events per patient (CGD dataset)", 
        ylab = "Number of Events", 
        col = "steelblue")
dev.off()

# NOTE: Step 4, the event-time plot for the first 10 patients, was removed.
# It was purely diagnostic, nothing downstream used it, and under
# R 4.5 / ggplot2 4.x it aborted the script with
#   Incompatible methods ("Ops.S7_object", "+.gg") for "+"
# before the CSV export could run.

############################################
# Step 5: Save Preprocessed Data as CSV
############################################

# Remove unused columns
cgd_data <- cgd_data %>%
  dplyr::select(id, time, gap_time, event, Z1, Z2, Z3, tau)

# View the structure of the final dataset
cat("Structure of the final CGD dataset:\n")
str(cgd_data)

# Save to a CSV file
write.csv(cgd_data, "data/cgd.csv", row.names = FALSE)
cat("The preprocessed CGD dataset with gap times has been saved as 'data/cgd.csv'.\n")


###############################################################################
# EXPORT FOR THE PYTHON DRIVERS
###############################################################################


export_df <- if (exists("data_cp")) data_cp else cgd_data
covariate_cols <- c("Z1","Z2","Z3")

# The Python loader keys on `delta`; `event` is retained for the R models.
export_df$delta <- as.integer(export_df$event)

# Fail loudly on non-positive gaps rather than filtering them away.
n_bad <- sum(export_df$gap_time <= 0, na.rm = TRUE)
if (n_bad > 0) {
  stop(sprintf(paste(
    "%d records have gap_time <= 0. The AFT models work on the log scale, so",
    "these cannot be represented. Decide explicitly how to handle them and",
    "apply the same decision to the Cox side, otherwise the two model",
    "families will be fitted on different rows."), n_bad))
}

keep <- c("id", "gap_time", "event", "delta", covariate_cols)
keep <- keep[keep %in% names(export_df)]
export_df <- export_df[, keep]

write.csv(export_df, "data/cgd.csv", row.names = FALSE)

cat("\n=== Integrity report ===\n")
cat("subjects:          ", length(unique(export_df$id)), "\n")
cat("records:           ", nrow(export_df), "\n")
cat("observed events:   ", sum(export_df$delta == 1), "\n")
cat("censored records:  ", sum(export_df$delta == 0), "\n")
cat("censoring fraction:",
    round(mean(export_df$delta == 0), 4), "\n")
cat("covariates:        ", paste(covariate_cols, collapse = ", "), "\n")
cat("written to:         data/cgd.csv\n")
cat("\nCheck these against the counts quoted in Section 5.1 of the\n")
cat("manuscript before running any experiment.\n")
