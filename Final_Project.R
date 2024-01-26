# Install necessary packages
install.packages(c("tidyverse", "caret", "pdp", "randomForest", "xgboost", "car", "gbm", "pROC"))

# Load libraries
library(tidyverse)
library(caret)
library(pdp)
library(randomForest)
library(xgboost)
library(car)
library(gbm)
library(pROC)

# Load the dataset
OJ <- read.csv(url("http://data.mishra.us/files/project/OJ_data.csv"))
OJ[2:14] <- lapply(OJ[2:14], as.numeric)
OJ$Purchase <- as.factor(OJ$Purchase)

#sample the first few rows
head(OJ)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(OJ$Purchase, p = 0.7, list = FALSE)
train_data <- OJ[trainIndex, ]
test_data <- OJ[-trainIndex, ]

# Fit logistic regression model
logistic_model <- glm(Purchase ~ ., data = train_data, family = "binomial")

# Extract coefficients from the logistic regression model
coefficients_table <- summary(logistic_model)$coefficients

# Filter coefficients related to predictor variables (excluding intercept)
predictor_coefficients <- coefficients_table[rownames(coefficients_table) != "(Intercept)", ]

# Display the predictor variables and their coefficients
print("1. Predictor Variables and Coefficients:")
print(predictor_coefficients)

# Assess variable significance
significant_variables <- predictor_coefficients[predictor_coefficients[, "Pr(>|z|)"] < 0.05, ]
print("\n2. Significant Variables:")
print(significant_variables)

# Predictions on the test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to predicted classes
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Ensure the predicted values have the same levels as the actual values
predicted_classes <- factor(predicted_classes, levels = levels(test_data$Purchase))

# Evaluate the model
model_performance <- caret::confusionMatrix(predicted_classes, as.factor(test_data$Purchase))
print("\n3. Model Performance:")
print(model_performance)

# Display specific recommendations based on coefficients
positive_influencers <- predictor_coefficients[predictor_coefficients[, "Estimate"] > 0, ]
negative_influencers <- predictor_coefficients[predictor_coefficients[, "Estimate"] < 0, ]

print("\n4. Recommendations for the Brand Manager:")
print("Factors positively influencing MM purchase:")
print(positive_influencers)

print("Factors negatively influencing MM purchase:")
print(negative_influencers)


# Build and Evaluate a Predictive Model for Sales Manager

# Fit logistic regression model on the entire dataset
full_logistic_model <- glm(Purchase ~ ., data = OJ, family = "binomial")

# Predictions on the test set for Sales Manager
full_predictions_sales_manager <- predict(full_logistic_model, newdata = test_data, type = "response")

# Convert probabilities to predicted classes
full_predicted_classes_sales_manager <- ifelse(full_predictions_sales_manager > 0.5, 1, 0)

# Ensure the predicted values have the same levels as the actual values
full_predicted_classes_sales_manager <- factor(full_predicted_classes_sales_manager, levels = levels(test_data$Purchase))

# Evaluate the model performance for Sales Manager
full_model_performance_sales_manager <- caret::confusionMatrix(full_predicted_classes_sales_manager, as.factor(test_data$Purchase))
print("\nSales Manager Model Performance:")
print(full_model_performance_sales_manager)

# Display specific recommendations based on coefficients for Sales Manager
full_positive_influencers_sales_manager <- coefficients_table[coefficients_table[, "Estimate"] > 0, ]
full_negative_influencers_sales_manager <- coefficients_table[coefficients_table[, "Estimate"] < 0, ]

print("\nRecommendations for the Sales Manager:")
print("Factors positively influencing MM purchase:")
print(full_positive_influencers_sales_manager)

print("Factors negatively influencing MM purchase:")
print(full_negative_influencers_sales_manager)

