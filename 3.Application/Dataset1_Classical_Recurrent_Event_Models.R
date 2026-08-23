
# Load necessary libraries
library(survival)
library(frailtypack)
library(reReg)
library(reda)
library(dplyr)
library(ggplot2)

# 1. Preprocess the `readmission` dataset
data(readmission, package = "frailtypack")

# Reformat the dataset into the required structure
# Recode chemo and sex for modeling
readmission$chemo <- ifelse(readmission$chemo == "Treated", 1, 0) # NonTreated = 0, Treated = 1
readmission$sex <- ifelse(readmission$sex == "Female", 1, 0) # Male = 0, Female = 1
readmission$dukes <- as.numeric(readmission$dukes)      # A-B = 1, C = 2, D = 3
readmission$charlson <- as.numeric(readmission$charlson) # Charlson comorbidity index

readmission <- readmission[order(readmission$id, readmission$t.stop), ]  # Sort by subject and time
last_index <- !duplicated(readmission$id, fromLast = TRUE)  # Identify last row per subject
readmission$event[last_index] <- 0  # Ensure the last event is censored

# Reformat and clean the dataset
formatted_data <- readmission %>%
  dplyr::select(id, t.start, t.stop, event, chemo, sex, dukes, time, death) %>%
  rename(
    start = t.start,
    stop = t.stop,
    gtime = time,  # Define `gtime` as gap times
    status = event,
    Z1 = chemo,
    Z2 = sex,
    Z3 = dukes
  ) %>%
  mutate(
    id = as.numeric(factor(id)),  # Ensure sequential IDs
    order = ave(id, id, FUN = seq_along)  # Add 'order' for stratification
  )

# Verify the formatted data
cat("Formatted data preview:\n")
print(head(formatted_data))
cat("Structure of the formatted data:\n")
str(formatted_data)
cat("Number of unique subjects:\n")
cat(length(unique(formatted_data$id)), "\n")

#################################
# Fit survival models
#################################

########################
# 1. Andersen–Gill model
########################
cat("\nFitting Andersen–Gill model...\n")
fit_AG <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3, 
  data = formatted_data
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
  data = formatted_data
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
  data = formatted_data
)
cat("LWYY Model Results:\n")
print(summary(fit_LWYY))

##################################
# 4. WLW model
##################################
cat("\nFitting WLW model...\n")
fit_wlw <- coxph(
  Surv(stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_data
)
cat("WLW Model Results:\n")
print(summary(fit_wlw))

##################################
# 5. PWP-TT model
##################################
cat("\nFitting PWP-TT model...\n")
fit_pwp_tt <- coxph(
  Surv(start, stop, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_data
)
cat("PWP-TT Model Results:\n")
print(summary(fit_pwp_tt))

##################################
# 6. PWP-GT model
##################################
cat("\nFitting PWP-GT model...\n")
fit_pwp_gt <- coxph(
  Surv(gtime, status) ~ Z1 + Z2 + Z3 * strata(order) + cluster(id),
  data = formatted_data
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
#formatted_data
