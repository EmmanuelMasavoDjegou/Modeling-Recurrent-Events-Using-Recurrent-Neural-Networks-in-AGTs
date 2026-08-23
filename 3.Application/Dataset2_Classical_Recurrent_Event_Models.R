# Load necessary libraries
library(survival)
library(dplyr)

# Load the CGD dataset
data(cgd, package = "survival")

# Display dataset dimensions and preview
cat("CGD dataset dimensions:\n")
print(dim(cgd))

cat("CGD dataset preview:\n")
print(head(cgd))

# Preprocessing the CGD dataset
formatted_cgd_data <- cgd %>%
  dplyr::select(id, tstart, tstop, status, enum, sex, age, treat) %>%  # Select relevant columns
  rename(
    start = tstart,
    stop = tstop,
    status = status,
    Z1 = treat,  # Use treatment as Z1 (categorical variable)
    Z2 = sex,    # Use sex as Z2 (categorical variable)
    Z3 = age     # Use age as Z3 (continuous variable)
  ) %>%
  mutate(
    # Encode Z1 and Z2 as factors or numeric (for modeling purposes)
    Z1 = ifelse(Z1 == "placebo", 0, 1),  # Placebo = 0, rIFN-g = 1
    Z2 = ifelse(Z2 == "male", 0, 1),     # Male = 0, Female = 1
    id = as.numeric(as.factor(id)),      # Ensure sequential IDs
    order = enum                         # Use enum for stratification/order
  )

cat("Formatted CGD data preview:\n")
print(head(formatted_cgd_data))
cat("Structure of the formatted CGD data:\n")
str(formatted_cgd_data)

#################################
# Fit survival models
#################################

########################
# 1. Andersen–Gill model
########################
cat("\nFitting Andersen–Gill model...\n")
fit_AG <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3,
  data = formatted_cgd_data
)
cat("Andersen–Gill Model Results:\n")
print(summary(fit_AG))

####################################
# 2. Frailty model (Gamma frailty)
####################################
cat("\nFitting Frailty model (Gamma frailty)...\n")
fit_frail <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3
  + frailty(id, distribution = "gamma"),
  data = formatted_cgd_data
)
cat("Frailty Model (Gamma) Results:\n")
print(summary(fit_frail))

############################################
# 3. LWYY (Proportional rates / means model)
############################################
cat("\nFitting LWYY model (cluster-based robust variance)...\n")
fit_LWYY <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3
  + cluster(id),
  data = formatted_cgd_data
)
cat("LWYY Model Results:\n")
print(summary(fit_LWYY))

##################################
# 4. WLW model
##################################
cat("\nFitting WLW model...\n")
fit_wlw <- coxph(
  Surv(stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_cgd_data
)
cat("WLW Model Results:\n")
print(summary(fit_wlw))

##################################
# 5. PWP-TT model
##################################
cat("\nFitting PWP-TT model...\n")
fit_pwp_tt <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_cgd_data
)
cat("PWP-TT Model Results:\n")
print(summary(fit_pwp_tt))

##################################
# 6. PWP-GT model
##################################
cat("\nFitting PWP-GT model...\n")
fit_pwp_gt <- coxph(
  Surv(stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_cgd_data
)
cat("PWP-GT Model Results:\n")
print(summary(fit_pwp_gt))

##################################
# Save model results and extract C-index
##################################
results_list <- list(
  Andersen_Gill = fit_AG,
  Frailty = fit_frail,
  LWYY = fit_LWYY,
  WLW = fit_wlw,
  PWP_TT = fit_pwp_tt,
  PWP_GT = fit_pwp_gt
)

cat("\nSaved all model results for future review.\n")

# Initialize a data frame to store C-index results
cindex_results <- data.frame(
  Model = c("Andersen-Gill", "Frailty", "LWYY", "WLW", "PWP-TT", "PWP-GT"),
  Concordance_Index = NA
)

# Extract Concordance Index for each model
for (i in seq_along(results_list)) {
  model_name <- names(results_list)[i]
  fit <- results_list[[i]]
  
  # Extract Concordance Index using summary()
  summary_fit <- summary(fit)
  concordance <- summary_fit$concordance[1]  # Extract C-index value (first value)
  
  # Save to the results data frame
  cindex_results$Concordance_Index[i] <- concordance
  
  # Print the Concordance Index for each model
  cat(paste(model_name, "C-index:", round(concordance, 4), "\n"))
}

# Display all C-index results
cat("\nC-index Results for All Models:\n")
print(cindex_results)