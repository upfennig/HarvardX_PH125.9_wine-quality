# Import libraries
library(tidyverse)
library(caret)
library(randomForest)
library(dplyr)
library(ggplot2)
library(stringr)
library(yaml)
library(knitr)

# Data set 

## Download file for white wine from UCI and remove temporary file
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
tmp_filename <- tempfile()
download.file(url, tmp_filename)
winequality <- read.csv(tmp_filename, sep =';')
file.remove(tmp_filename)

## Set number of significant digits
options(digits = 4)

## Validate imported data file   
dim(winequality)
str(winequality)
sum(is.na(winequality))

## Review sample data 
head(winequality,3)


# Method and analysis: Data exploration

## Descriptive summary
summary(winequality)

## Distribution of quality across data set
summary(winequality$quality)
winequality %>% group_by(quality) %>% summarize(n = n(), quality = quality[1]) 
winequality %>% 
  ggplot(aes(quality)) + 
  geom_histogram() +
  ggtitle("Number of wines per quality rating")

winequality %>% 
  ggplot(aes(y=quality)) + 
  geom_boxplot()

## Categorize quality into 3 groups 
winequality <- winequality %>% mutate(quality_cat = factor(case_when(
  quality %in% c(3,4) ~ "1-low",
  quality %in% c(5,6) ~ "2-medium",
  quality > 6 ~ "3-high"))) %>% 
  select(-quality)


## Review ingredients per quality category
winequality %>% gather(ingredients, percentage, -quality_cat) %>%
  ggplot(aes(quality_cat, percentage, fill = quality_cat)) +
  geom_boxplot() +
  facet_wrap(~ingredients, scales = "free")



# Methods and analysis: modeling 

## Split data into test and train data using 20% of the data for validation
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = winequality$quality_cat, times = 1, p = 0.2, list = FALSE)
train_set <- winequality[-test_index,]
test_set <- winequality[test_index,]


## default kNN model using caret package
train_knn <- train(quality_cat ~ ., method = "knn", data = train_set)
y_hat_knn <- predict(train_knn, test_set, type = "raw")
accuracy_knn <- confusionMatrix(y_hat_knn, test_set$quality_cat)$overall[["Accuracy"]]
accuracy_knn

ggplot(train_knn, highlight= TRUE)

## accuracy results table 
quality_results <- tibble(method = "Default kNN model", Accuracy = accuracy_knn)


## tune kNN model to find the right k
train_knn_2 <- train(quality_cat ~ ., method = "knn", 
                   data = train_set,
                   tuneGrid = data.frame(k = seq(9, 50, 3)))
train_knn_2$bestTune
y_hat_knn_2 <- predict(train_knn_2, test_set, type = "raw")
accuracy_knn_2 <- confusionMatrix(y_hat_knn_2, test_set$quality_cat)$overall[["Accuracy"]]
accuracy_knn_2

ggplot(train_knn_2, highlight= TRUE)

## accuracy results table 
quality_results <- bind_rows(quality_results,
                             tibble(method="Optimized kNN model (k=42)", Accuracy = accuracy_knn_2))
quality_results %>% knitr::kable()


## Classification tree
train_rpart <- train(quality_cat ~ ., method = "rpart", data = train_set)
y_hat_rpart <- predict(train_rpart, test_set, type = "raw")
accuracy_rpart <- confusionMatrix(y_hat_rpart, test_set$quality_cat)$overall[["Accuracy"]]
accuracy_rpart

## decision tree plot
plot(train_rpart$finalModel, margin=0.1)
text(train_rpart$finalModel, cex=0.75)

## accuracy results table 
quality_results <- bind_rows(quality_results,
                             tibble(method="Classification tree", Accuracy = accuracy_rpart))
quality_results %>% knitr::kable()

## variable importance
varImp(train_rpart)

## Optimize classification tree related to CP value (complexity parameter)
# train_rpart_2 <- train(quality_cat ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0.0, 0.08, len = 20)), data = train_set)
# confusionMatrix(predict(train_rpart_2, test_set),
#                test_set$quality_cat)$overall["Accuracy"]
# plot(train_rpart_2)
# optimization excluded because of limited effectiveness


## randomForest 
train_rf <- randomForest(quality_cat ~ ., data = train_set)
accuracy_rf <- confusionMatrix(predict(train_rf, test_set),
                test_set$quality_cat)$overall["Accuracy"]
accuracy_rf


## accuracy results table 
quality_results <- bind_rows(quality_results,
                             tibble(method="Random forest", Accuracy = accuracy_rf))

## variable importance rf
varImpPlot(train_rf)

## accuracy results table in results section
quality_results %>% knitr::kable()

## randomForest_optimized; optimizing the minimum number of data points in a node
# nodesize <- seq(1, 51, 10)
# acc <- sapply(nodesize, function(ns){
#  train(quality_cat ~ ., method = "rf", data = train_set,
#        tuneGrid = data.frame(mtry = 2),
#        nodesize = ns)$results$Accuracy
#})
# train_rf_2 <- randomForest(quality_cat ~ ., data = train_set,
#                           nodesize = nodesize[which.max(acc)])

# confusionMatrix(predict(train_rf_2, test_set),
#                test_set$quality_cat)$overall["Accuracy"]
# optimization excluded because of limited effectiveness






