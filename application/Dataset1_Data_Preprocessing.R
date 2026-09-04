###############################################################################
# Dataset 1: colorectal cancer readmission (frailtypack::readmission)
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
# there:  Rscript application/Dataset1_Data_Preprocessing.R
if (!dir.exists("data"))    dir.create("data", recursive = TRUE)
if (!dir.exists("results")) dir.create("results", recursive = TRUE)

# Load necessary libraries
library(frailtypack)
library(survival)
library(reReg)
library(reda)
library(dplyr)
library(ggplot2)

# Load the dataset
data(readmission, package = "frailtypack")
head(readmission)
?readmission

# Check the structure of the dataset to understand its structure
str(readmission)

# Subset the data by treatment group (chemo = 1 for non-treated, 2 for treated)
treated_data <- subset(readmission, chemo == "Treated")
non_treated_data <- subset(readmission, chemo == "NonTreated")

# Count the number of events per patient for each group
events_per_patient_treated <- table(treated_data$id)
events_per_patient_non_treated <- table(non_treated_data$id)

# Summary of events per patient for treated and non-treated
summary(as.numeric(events_per_patient_treated))
summary(as.numeric(events_per_patient_non_treated))

# Find the patient ID with the maximum number of events in each group
patient_with_max_events_treated <- names(which.max(events_per_patient_treated))
patient_with_max_events_non_treated <- names(which.max(events_per_patient_non_treated))

# Find the number of events for the patient with the maximum events in each group
max_events_treated <- max(events_per_patient_treated)
max_events_non_treated <- max(events_per_patient_non_treated)

# Print the result for both groups
cat("Treated - Patient ID with the highest number of events:", patient_with_max_events_treated, "\n")
cat("Treated - Number of events for this patient:", max_events_treated, "\n")

cat("Non-Treated - Patient ID with the highest number of events:", patient_with_max_events_non_treated, "\n")
cat("Non-Treated - Number of events for this patient:", max_events_non_treated, "\n")


# Export histogram and boxplot as PNG files

# Create the histogram and boxplot for treated and non-treated groups
png("histograms_treated_non_treated.png", width = 800, height = 400) # Set the file path and dimensions
par(mfrow = c(1, 2)) # Display side by side
hist(as.numeric(events_per_patient_treated), breaks = 30, 
     main = "Treated", 
     xlab = "Number of Events", 
     col = "firebrick")
hist(as.numeric(events_per_patient_non_treated), breaks = 30, 
     main = "Non-Treated", 
     xlab = "Number of Events", 
     col = "orange")
dev.off() # Close the device

# Export boxplot for treated and non-treated groups
png("boxplots_treated_non_treated.png", width = 800, height = 400) # Set the file path and dimensions
par(mfrow = c(1, 2)) # Display side by side
boxplot(as.numeric(events_per_patient_treated), 
        main = "Treated", 
        ylab = "Number of Events", 
        col = "firebrick")
boxplot(as.numeric(events_per_patient_non_treated), 
        main = "Non-Treated", 
        ylab = "Number of Events", 
        col = "orange")
dev.off() # Close the device

# Subset data for the first 10 patients
first_10_patients <- readmission %>%
  filter(id %in% unique(readmission$id)[1:10])

# Assign colors based on treatment status
first_10_patients <- first_10_patients %>%
  mutate(color = ifelse(chemo == "Treated", "firebrick", "orange"))

# Create the plot
ggplot(first_10_patients, aes(x = t.stop, y = factor(id))) +
  geom_point(aes(color = color, shape = factor(event)), size = 2) +  
  scale_shape_manual(values = c(4,1)) +  # 3 for censored (cross), 1 for event (circle)
  scale_color_identity() +  # Directly use assigned colors
  labs(x = "Event Time (Days)", 
       y = "Patient ID") +
  theme_minimal() +
  theme(legend.position = "none",  
        axis.text.x = element_text(angle = 45, hjust = 1))

# Convert categorical variables to numeric encoding
readmission$chemo <- ifelse(readmission$chemo == "Treated", 1, 0) # 0 = NonTreated, 1 = Treated
readmission$sex <- ifelse(readmission$sex == "Female", 1, 0) # 0 = Male, 1 = Female
readmission$dukes <- as.numeric(readmission$dukes)      # A-B = 1, C = 2, D = 3
readmission$charlson <- as.numeric(readmission$charlson) # 0 = 1, 1-2 = 2, 3 = 3

# Check for missing values
colSums(is.na(readmission))

# Ensure last event for each subject is censored (event = 0)
readmission <- readmission[order(readmission$id, readmission$t.stop), ]  # Sort by subject and time
last_index <- !duplicated(readmission$id, fromLast = TRUE)  # Identify last row per subject
readmission$event[last_index] <- 0  # Set last event as censored

# Check for multiple terminal events (death = 1) per subject
multiple_deaths <- table(readmission$id[readmission$death == 1])
problem_ids <- names(multiple_deaths[multiple_deaths > 1])

# Remove all records for subjects with multiple death events
readmission <- readmission[!(readmission$id %in% problem_ids), ]

# Reorder the id variable to maintain sequential numbering
readmission$id <- as.integer(factor(readmission$id))

# Print first few rows to verify
head(readmission)
colSums(is.na(readmission))

data_cp <- readmission %>%
  dplyr::select(id, t.stop, time, event, chemo, sex, dukes, charlson)


# Create a new column for censored time, where the last t.stop time is assigned for each patient
data_cp <- data_cp %>%
  group_by(id) %>%
  mutate(censored_time = max(t.stop)) %>%
  ungroup()


# Rename the columns as per your requirements
data_cp <- data_cp %>%
  dplyr::select(id, t.stop, time, event, chemo, sex, dukes, charlson, censored_time) %>%
  rename(
    id = id,
    time = t.stop,
    gap_time = time,
    Z1 = chemo,
    Z2 = sex,
    Z3 = dukes,
    Z4 = charlson,
    tau = censored_time,
  )

# Remove rows where gap_time equals zero
data_cp <- data_cp %>%
  filter(gap_time != 0)

# Reorder the ids from 1 to n
data_cp <- data_cp %>%
  mutate(id = as.numeric(factor(id)))

# View the first few rows of the renamed dataset
head(data_cp)
str(data_cp)
length(unique(data_cp$id))

# Save the filtered dataset to a CSV file
write.csv(data_cp, "data/crc.csv", row.names = FALSE)

###############################################################################
# EXPORT FOR THE PYTHON DRIVERS
###############################################################################


export_df <- if (exists("data_cp")) data_cp else cgd_data
covariate_cols <- c("Z1","Z2","Z3","Z4")

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

write.csv(export_df, "data/crc.csv", row.names = FALSE)

cat("\n=== Integrity report ===\n")
cat("subjects:          ", length(unique(export_df$id)), "\n")
cat("records:           ", nrow(export_df), "\n")
cat("observed events:   ", sum(export_df$delta == 1), "\n")
cat("censored records:  ", sum(export_df$delta == 0), "\n")
cat("censoring fraction:",
    round(mean(export_df$delta == 0), 4), "\n")
cat("covariates:        ", paste(covariate_cols, collapse = ", "), "\n")
cat("written to:         data/crc.csv\n")
cat("\nCheck these against the counts quoted in Section 5.1 of the\n")
cat("manuscript before running any experiment.\n")
