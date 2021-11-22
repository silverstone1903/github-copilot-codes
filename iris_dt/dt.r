library(randomForest)
library(nnet)
library(xgboost)
library(lightgbm)
library(caret)
library(catboost)
data(iris)

#### train parameters
seed = 2021
train_ratio = 0.8

#### train test split 
set.seed(seed)
splitter <- createDataPartition(iris$Species, list = F, p = train_ratio)
train <- iris[splitter, ]
test <- iris[-splitter, ]

# Binary Logistic Regression
# lr <- glm(Species ~ ., data = train, family = binomial)
# lr_preds <- predict(lr, test)

# Multinomial Logistic Regression
set.seed(seed)
lr <- multinom(Species ~ ., data = train)
lr_preds <- predict(lr, test)
confusionMatrix(lr_preds, test$Species)

# Random Forest
set.seed(seed)
rf <- randomForest(Species ~ ., data = train, ntree = 100, seed = seed)
rf_pred <- predict(rf, test, type = "response") 
confusionMatrix(rf_pred, test$Species)

# XGBoost
set.seed(seed)
xgb <- xgboost(data =  as.matrix(train[,1:4]), label = as.integer(train$Species)-1, max.depth = 2, eta = 0.1, nthread = 12, nrounds = 10, objective = "multi:softmax", num_class = 3)
xgb_pred <- predict(xgb, xgb.DMatrix(as.matrix(test[,1:4])), type = "response")
confusionMatrix(as.factor(xgb_pred), as.factor(as.integer(test$Species)-1))

# LightGBM
dtrain <- lgb.Dataset(data = as.matrix(train[, 1:4]), label = as.integer(train$Species)-1)
params <- list(
    objective = "multiclass", 
    metric = "multi_error", 
    num_class = 3, 
    min_data = 1, 
    learning_rate = 0.1
)

lgb_model <- lgb.train(params, dtrain, 100)
lgbm_pred <- predict(lgb_model, data = as.matrix(test[, 1:4]), reshape = TRUE)
lgb_max <- apply(as.array(lgbm_pred), 1, which.max)
confusionMatrix(as.factor(lgb_max-1), as.factor(as.integer(test$Species)-1))

# catboost
set.seed(seed)
train_pool <- catboost.load_pool(data = train[, 1:4], label = as.integer(train$Species)-1)
test_pool <- catboost.load_pool(data = test[, 1:4], label = as.integer(test$Species)-1)

params <- list(iterations = 100,
                #loss_function = "MultiClass",
                metric_period = 20,
                random_seed = seed)

model <- catboost.train(train_pool, test_pool, params)
cb_preds <- catboost.predict(model, test_pool, prediction_type = "Class")
confusionMatrix(as.factor(cb_preds), as.factor(as.integer(test$Species)-1))
