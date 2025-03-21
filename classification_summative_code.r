install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3tuning")
install.packages("mlr3viz")
install.packages("ranger")
install.packages("pROC")
install.packages("precrec")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("corrplot")

# Load libraries
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3viz)
library(ranger)
library(pROC)
library(precrec)
library(rpart)
library(rpart.plot)
library(corrplot)


# Reading the data
bpl <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv",
                header=TRUE)


numeric_data <- bpl[, sapply(bpl, is.numeric)]

## Studying the correlation
# correlation matrix
cor_matrix <- cor(numeric_data)

# Plot the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)


## Splitting the training and the testing data

## 80% of the sample size
smp_size <- floor(0.80 * nrow(bpl))

## set the seed to make the partition reproducible
set.seed(42)
train_ind <- sample(seq_len(nrow(bpl)), size = smp_size)

train <- bpl[train_ind, ]
test <- bpl[-train_ind, ]


## Logistic Regression

# Fit logistic regression model
logit_model <- glm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                   data = train, 
                   family = binomial(link = "logit"))

# Display the model summary
summary(logit_model)

# Refit logistic regression model after removing the non-significant predictors
logit_model <- glm(Personal.Loan ~ Income + Family + CCAvg + Education + Securities.Account + CD.Account + Online + CreditCard, 
                   data = train, 
                   family = binomial(link = "logit"))

# Display the model summary
summary(logit_model)

# Probabilities
predicted_probabilities <- predict(logit_model, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = train$Personal.Loan)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Precision:", precision))

# Grid search parameter grid
param_grid <- expand.grid(
  minsplit = c(10, 20, 30),
  minbucket = c(5, 10, 15),
  cp = c(0.001, 0.01, 0.1),
  maxdepth = c(5, 10, 15) 
)

# Initialize variables to store results
results <- data.frame()


for (i in 1:nrow(param_grid)) {
  minsplit <- param_grid$minsplit[i]
  minbucket <- param_grid$minbucket[i]
  cp <- param_grid$cp[i]
  maxdepth <- param_grid$maxdepth[i]
  
  # Fitting the decision tree model
  tree_model <- rpart(Personal.Loan ~ Income + Family + CCAvg + Education + Securities.Account + CD.Account + Online + CreditCard, 
                      data = train, 
                      method = "class", 
                      control = rpart.control(minsplit = minsplit, minbucket = minbucket, cp = cp, maxdepth = maxdepth))
  
  # Predictions
  predicted_classes <- predict(tree_model, train, type = "class")
  
  # Accuracy
  accuracy <- mean(predicted_classes == train$Personal.Loan)
  
  results <- rbind(results, data.frame(minsplit, minbucket, cp, maxdepth, accuracy))
}

print(results)

# Extract the best hyperparameters
best_params <- results[which.max(results$accuracy), ]
print(best_params)


### Training the tree model with the best hyper parameters
final_tree_model <- rpart(Personal.Loan ~ Income + Family + CCAvg + Education + Securities.Account + CD.Account + Online + CreditCard, 
                          data = train, 
                          method = "class", 
                          control = rpart.control(minsplit = best_params$minsplit, 
                                                  minbucket = best_params$minbucket, 
                                                  cp = best_params$cp, 
                                                  maxdepth = best_params$maxdepth))

# Visualizing the final tree
rpart.plot(final_tree_model, type = 4, extra = 104, box.palette = "auto", nn = TRUE)

# Predictions
predicted_classes <- predict(final_tree_model, train, type = "class")

# Confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = train$Personal.Loan)
print(conf_matrix)

# Calculate metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

## Comparing the performance of logistic regression and decision tree on testing data


# Probabilities
predicted_probabilities <- predict(logit_model, test, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

print("Logistic Regression")
# Confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = test$Personal.Loan)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Precision:", precision))

recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

# Predictions
predicted_classes <- predict(final_tree_model, test, type = "class")

# Confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = test$Personal.Loan)
print(conf_matrix)

# Performance metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

print("\nDecision Tree")
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))


### Random Forests

# Converting Personal.Loan to factor
train$Personal.Loan <- as.factor(train$Personal.Loan)

# Classification task
task <- TaskClassif$new(id = "PersonalLoan", backend = train, target = "Personal.Loan")


learner <- lrn("classif.ranger", predict_type = "prob")
learner$train(task)

# Predictions
predictions <- learner$predict(task)

# Confusion matrix
conf_matrix <- predictions$confusion
print(conf_matrix)

# Performance metrics
accuracy <- predictions$score(msr("classif.acc"))
precision <- predictions$score(msr("classif.precision"))
recall <- predictions$score(msr("classif.recall"))
f1_score <- predictions$score(msr("classif.fbeta", beta = 1))

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

# AUC
auc_value <- predictions$score(msr("classif.auc"))
print(paste("AUC:", auc_value))

# Plot ROC curve
autoplot(predictions, type = "roc")

# Converting Personal.Loan to factor
test$Personal.Loan <- as.factor(test$Personal.Loan)

test_task <- TaskClassif$new(id = "PersonalLoanTest", backend = test, target = "Personal.Loan")

# Test predictions
test_predictions <- learner$predict(test_task)
print(test_predictions)

# Confusion matrix
conf_matrix <- test_predictions$confusion
print(conf_matrix)

# Performance metrics
accuracy <- test_predictions$score(msr("classif.acc"))
precision <- test_predictions$score(msr("classif.precision"))
recall <- test_predictions$score(msr("classif.recall"))
f1_score <- test_predictions$score(msr("classif.fbeta", beta = 1))

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

# AUC
auc_value <- test_predictions$score(msr("classif.auc"))
print(paste("AUC:", auc_value))

# ROC curve
autoplot(test_predictions, type = "roc")

## Hyper parameter tuning for the random forest algorithm

# Converting Personal.Loan to factor
train$Personal.Loan <- as.factor(train$Personal.Loan)
test$Personal.Loan <- as.factor(test$Personal.Loan)

train_task <- TaskClassif$new(id = "PersonalLoanTrain", backend = train, target = "Personal.Loan")
test_task <- TaskClassif$new(id = "PersonalLoanTest", backend = test, target = "Personal.Loan")

# Grid search hyperparameter grid
param_grid <- expand.grid(
  mtry = c(2, 3, 4),                # Number of variables to try at each split
  min.node.size = c(5, 10, 15),     # Minimum node size
  num.trees = c(300, 500, 700)      # Number of trees
)

# Results variable
results <- data.frame()

# Grid search
for (i in 1:nrow(param_grid)) {
  # Extract hyperparameters
  mtry <- param_grid$mtry[i]
  min.node.size <- param_grid$min.node.size[i]
  num.trees <- param_grid$num.trees[i]
  
  
  learner <- lrn("classif.ranger", 
                 mtry = mtry, 
                 min.node.size = min.node.size, 
                 num.trees = num.trees, 
                 predict_type = "prob")  
  
  # Training the model
  learner$train(train_task)
  
  # Making predictions on the training data 
  predictions <- learner$predict(train_task)  
  
  # AUC
  auc_value <- predictions$score(msr("classif.auc"))
  
  results <- rbind(results, data.frame(mtry, min.node.size, num.trees, auc_value))
}

print(results)

# Extracting the best hyperparameters
best_params <- results[which.max(results$auc_value), ]
print(best_params)


## Training and testing the best model
final_learner <- lrn("classif.ranger", 
                     mtry = best_params$mtry, 
                     min.node.size = best_params$min.node.size, 
                     num.trees = best_params$num.trees, 
                     predict_type = "prob")  # Set predict_type here

final_learner$train(train_task)

# Predictions on the training data
train_predictions <- final_learner$predict(train_task)

# Predictions on the testing data
test_predictions <- final_learner$predict(test_task) 

# Confusion matrix
train_conf_matrix <- train_predictions$confusion
print(train_conf_matrix)

# Performance metrics
train_accuracy <- train_predictions$score(msr("classif.acc"))
train_precision <- train_predictions$score(msr("classif.precision"))
train_recall <- train_predictions$score(msr("classif.recall"))
train_f1_score <- train_predictions$score(msr("classif.fbeta", beta = 1))

print(paste("Training Accuracy:", train_accuracy))
print(paste("Training Precision:", train_precision))
print(paste("Training Recall:", train_recall))
print(paste("Training F1-Score:", train_f1_score))

# AUC
auc_value <- test_predictions$score(msr("classif.auc"))
print(paste("AUC:", auc_value))

# ROC curve
autoplot(test_predictions, type = "roc")

# Confusion matrix
conf_matrix <- test_predictions$confusion
print(conf_matrix)

# Performance metrics
accuracy <- test_predictions$score(msr("classif.acc"))
precision <- test_predictions$score(msr("classif.precision"))
recall <- test_predictions$score(msr("classif.recall"))


print(paste("Testing Accuracy:", accuracy))
print(paste("Testing Precision:", precision))
print(paste("Testing Recall:", recall))

