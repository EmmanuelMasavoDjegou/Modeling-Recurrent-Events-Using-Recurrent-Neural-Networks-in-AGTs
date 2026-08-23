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
write.csv(data_cp, "data_cp.csv", row.names = FALSE)