# install packages
#install.packages('glmnet')
#install.packages('caret')
#install.packages('dplyr')

# load libraries
library(glmnet)
library(caret)
library(dplyr)

# load dataset
load("./trabalhosalarios.RData")
data <- as.data.frame(trabalhosalarios[, !names(trabalhosalarios) %in% c('earns')])

# set seed
set.seed(123)

# create index for data partition
index <- sample(1:nrow(data), 0.8 * nrow(data))

# create train database
train <- data[index,]

# create test database
test <- data[-index,]

# standardize variables
numeric_columns <- c('husage', 'husearns', 'huseduc', 'hushrs',
                     'age', 'educ', 'exper')
pre_proc_val <- preProcess(train[,numeric_columns],
                           method = c("center", "scale"))
train[,numeric_columns] <- predict(pre_proc_val, train[,numeric_columns])
test[,numeric_columns] <- predict(pre_proc_val, test[,numeric_columns])

# create used variables list
columns <- c('husage', 'husunion', 'husearns',
             'huseduc', 'husblck', 'hushisp',
             'hushrs', 'kidge6', 'age',
             'black', 'educ', 'hispanic',
             'union', 'exper', 'kidlt6', 'lwage')

# generate dummies
dummies <- dummyVars(lwage ~ husage + husunion + husearns
                     + huseduc + husblck + hushisp
                     + hushrs + kidge6 + age
                     + black + educ + hispanic 
                     + union + exper + kidlt6,
                     data = data[,columns])
train_dummies <- predict(dummies, newdata = train[,columns])
test_dummies <- predict(dummies, newdata = test[,columns])

# define x_train as training matrix for independent variables
x_train <- as.matrix(train_dummies)

# define y_train as training matrix for dependent variable
y_train <- train$lwage

# define x_test as test matrix for independent variables
x_test <- as.matrix(test_dummies)

# define y_test as test matrix for dependent variable
y_test <- test$lwage

# adjust models configuration
lambdas <- 10 ^ seq(2, -3, by = -.1)
ridge_lambdas <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambdas)
lasso_lambdas <- cv.glmnet(x_train, y_train, alpha = 1, lambda = lambdas,
                           standardize = TRUE, nfolds = 5)
best_ridge_lambda <- ridge_lambdas$lambda.min
best_lasso_lambda <- lasso_lambdas$lambda.min

elastic_train_control <- 
  trainControl(method = "repeatedcv", number = 10, repeats = 5,
               search = "random", verboseIter = TRUE)

train_formula <- as.formula(lwage ~ husage + husunion + husearns
                           + huseduc + husblck + hushisp
                           + hushrs + kidge6 + age
                           + black + educ + hispanic 
                           + union + exper + kidlt6)

# train models
ridge <- glmnet(x_train, y_train, nlambda = 25, alpha = 0,
               family = 'gaussian', lambda = best_ridge_lambda)
lasso <- glmnet(x_train, y_train, alpha = 1, lambda = best_lasso_lambda,
               standardize = TRUE)
elastic <- train(train_formula,
                 data = train,
                 method = "glmnet",
                 tuneLength = 10,
                 trControl = elastic_train_control)

# define method to evaluate the results
evaluate_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true) ^ 2)
  SST <- sum((true - mean(true)) ^ 2)
  R_square <- 1 - SSE / SST
  RMSE <- sqrt(SSE / nrow(df))
  
  return(data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  ))
}

# make predictions for the training database
predictions.ridge.train <- predict(ridge, s = best_ridge_lambda, newx = x_train)
predictions.lasso.train <- predict(lasso, s = best_lasso_lambda, newx = x_train)
predictions.elastic.train <- predict(elastic, x_train)

# evaluate train predictions
results.ridge.train <- evaluate_results(y_train, predictions.ridge.train, train)
results.lasso.train <- evaluate_results(y_train, predictions.lasso.train, train)
results.elastic.train <- evaluate_results(y_train, predictions.elastic.train, train)

# make predictions for the test database
predictions.ridge.test <- predict(ridge, s = best_ridge_lambda, newx = x_test)
predictions.lasso.test <- predict(lasso, s = best_lasso_lambda, newx = x_test)
predictions.elastic.test <- predict(elastic, x_test)

# evaluate test predictions
results.ridge.test <- evaluate_results(y_test, predictions.ridge.test, test)
results.lasso.test <- evaluate_results(y_test, predictions.lasso.test, test)
results.elastic.test <- evaluate_results(y_test, predictions.elastic.test, test)

# insert new data predicted
make_pre_proc <- function (value, colname) {
  (value - pre_proc_val[["mean"]][[colname]]) / pre_proc_val[["std"]][[colname]]
}

new_data <- as.matrix(data.frame(
  husage = make_pre_proc(40, 'husage'),
  husunion = 0,
  husearns = make_pre_proc(600, 'husearns'),
  huseduc = make_pre_proc(13, 'huseduc'),
  husblck = 1,
  hushisp = 0,
  hushrs = make_pre_proc(40, 'hushrs'), 
  kidge6 = 1,
  age = make_pre_proc(38, 'age'),
  black = 0,
  educ = make_pre_proc(13, 'educ'),
  hispanic = 1,
  union = 0,
  exper = make_pre_proc(18, 'exper'),
  kidlt6 = 1
))

new_predict.ridge <- predict(ridge, s = best_ridge_lambda, new_data)
new_predict.lasso <- predict(lasso, s = best_lasso_lambda, new_data)
new_predict.elastic <- predict(elastic, new_data)

normalize_wage <- function(lwage) {
  return((lwage * sd(train$lwage)) + mean(train$lwage))
}

results.ridge.new <- normalize_wage(new_predict.ridge)
results.lasso.new <- normalize_wage(new_predict.lasso)
results.elastic.new <- normalize_wage(new_predict.elastic)

n <- nrow(train) # sample size
s <- sd(data$lwage) # standard deviation
dam <- s/sqrt(n) # sample mean distribution

# confidence intervals
results.ridge.new.ci.lower <- results.ridge.new + (qnorm(0.025)) * dam
results.ridge.new.ci.upper <- results.ridge.new - (qnorm(0.025)) * dam
results.lasso.new.ci.lower <- results.lasso.new + (qnorm(0.025)) * dam
results.lasso.new.ci.upper <- results.lasso.new - (qnorm(0.025)) * dam
results.elastic.new.ci.lower <- results.elastic.new + (qnorm(0.025)) * dam
results.elastic.new.ci.upper <- results.elastic.new - (qnorm(0.025)) * dam

# create tables with results
tb_stats <- data.frame(
  c('ridge', 'lasso', 'elastic'),
  c(results.ridge.train$Rsquare, results.lasso.train$Rsquare, results.elastic.train$Rsquare),
  c(results.ridge.train$RMSE, results.lasso.train$RMSE, results.elastic.train$RMSE),
  c(results.ridge.test$Rsquare, results.lasso.test$Rsquare, results.elastic.test$Rsquare),
  c(results.ridge.test$RMSE, results.lasso.test$RMSE, results.elastic.test$RMSE)
)
             
colnames(tb_stats) <- c('model', 'r2 (train)', 'rmse (train)', 'r2 (test)', 'rmse (test)')
View(tb_stats)

tb_predict <- data.frame(
  c('ridge', 'lasso', 'elastic'),
  c(results.ridge.new, results.lasso.new, results.elastic.new),
  c(results.ridge.new.ci.lower, results.lasso.new.ci.lower, results.elastic.new.ci.lower),
  c(results.ridge.new.ci.upper, results.lasso.new.ci.upper, results.elastic.new.ci.upper)
)

colnames(tb_predict) <- c('model', 'predict', 'CI-lower', 'CI-upper')
View(tb_predict)
