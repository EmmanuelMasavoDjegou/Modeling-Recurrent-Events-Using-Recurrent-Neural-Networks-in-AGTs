# Load necessary libraries
library(frailtypack)
library(survival)
library(dplyr)
library(ggplot2)

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

############################################
# Step 4: Event Time Visualization for the First 10 Patients
############################################

# Subset data for the first 10 patients
first_10_patients_cgd <- cgd_data %>%
  filter(id %in% unique(cgd_data$id)[1:10])

# Assign colors based on treatment (Z1)
# Orange for placebo, firebrick for rIFN-g
first_10_patients_cgd <- first_10_patients_cgd %>%
  mutate(color = ifelse(Z1 == 1, "firebrick", "orange"))

# Create a plot
ggplot(first_10_patients_cgd, aes(x = time, y = factor(id))) +
  geom_point(aes(color = color, shape = factor(event)), size = 2) +
  scale_shape_manual(values = c(4, 1)) +  # 4 for censored, 1 for event
  scale_color_identity() +  # Use assigned colors
  labs(x = "Event Time (Days)", 
       y = "Patient ID") +
  theme_minimal() +
  theme(legend.position = "none",  
        axis.text.x = element_text(angle = 45, hjust = 1))

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
write.csv(cgd_data, "cgd_preprocessed.csv", row.names = FALSE)
cat("The preprocessed CGD dataset with gap times has been saved as 'cgd_preprocessed.csv'.\n")
