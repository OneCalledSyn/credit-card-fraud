#Load desired packages
library(readr)
library(dplyr)
library(randomForest)
library(ggplot2)
library(Hmisc)
library(caret)
library(MLmetrics)
library(e1071)
library(party)


#Allows for model reproducibility
set.seed(1234)

#Import the credit card data from local
creditData <- read_csv("C:/Users/OneCalledSyn/Desktop/creditcard.csv")

#Data fram has 284,807 observations with 31 variables
glimpse(creditData)

#Make Class a factor
creditData$Class <- factor(creditData$Class)

#Proof Class is a factor with two levels, 0 and 1
str(creditData$Class)

#Partition data set into the training set and the test set (~70/30 split)
train <- creditData[1:7000, ]
test <- creditData[7001:10000, ]

#Check to see how many fraudulent transactions are in the training set
train %>%
  select(Class) %>%
  group_by(Class) %>%
  summarise(count= n()) %>%
  glimpse

#Now do the same for the test set
test %>%
  select(Class) %>%
  group_by(Class) %>%
  summarise(count= n()) %>%
  glimpse

#Build a random forest model with all predictors included
rf_model <- randomForest(Class ~ ., data = train, proximity = TRUE)
rf_model

#Test the model by using it to predict the test set
test$predicted <- predict(rf_model, test)

#Examine how well the prediction performed (CI, TP/FP/TN/FN, p-value, etc.)
confusionMatrix(test$Class, test$predicted)

#F1 score is the harmonic mean of precision and recall
#F1 score: Measure of the accuracy of a binary classification test
#Precision: Fraction of relevant observations over returned instances
#Recall: Fraction of relevant observations that were returned
#Harmonic mean: Reciprocal of the arithmetic mean of the reciprocals of the observations
F1_stat <- F1_Score(test$Class, test$predicted)
F1_stat 
#F1 = 0.9996651

#Which predictors matter the most? Can we improve the model in speed/precision by using a diferent set of predictors?
#Function specifically designed for showing variable importance for Random FOrests
varImpPlot(rf_model, sort = TRUE, n.var = 15, main = "How important are the variables?")

#Gini impurity: How often would a randomly chosen element from the set be incorrectly labeled 
#if it was randomly labeled according to the distribution of labels in the subset
#Measure of node impurity, the greater the mean decrease, the more important the variable is

#Try a new model only using the most important variable, V12
rf_model_V12 <- randomForest(Class ~ V12, data = train)

test$predictedV12 <- predict(rf_model_V12, test)

F1_stat_V12 <- F1_Score(test$Class, test$predictedV12)
F1_stat_V12 

#F1 = 0.9994976, so a lower score than using all predictors, but faster runtime

#Try a model with the top 3 predictors
rf_model_top3 <- randomForest(Class ~ V12 + V14 + V11, data = train)

test$predictedtop3 <- predict(rf_model_top3, test)

F1_stat_top3 <- F1_Score(test$Class, test$predictedtop3)
F1_stat_top3 

#F1 = 0.9994976 (same as all predictor model out to 15 decimal places)
#So we are able to get an equally good F1 score with only three predictors

#Let's go back and try the top two 
rf_model_top2 <- randomForest(Class ~ V12 + V14, data = train)

test$predictedtop2 <- predict(rf_model_top2, test)

F1_stat_top2 <- F1_Score(test$Class, test$predictedtop2)
F1_stat_top2

#F1 = 0.9994976 Still the same F1 score as the top 3 and the all-inclusive model
#The top two model is objectively better than the top 3 and all-inclusive model

#Let's go the other way and test more combinations
rf_model_top4 <- randomForest(Class ~ V12 + V14 + V11 + V10, data = train)

test$predictedtop4 <- predict(rf_model_top4, test)

F1_stat_top4 <- F1_Score(test$Class, test$predictedtop4)
F1_stat_top4 


rf_model_top5 <- randomForest(Class ~ V12 + V14 + V11 + V10 + V17, data = train)

test$predictedtop5 <- predict(rf_model_top5, test)

F1_stat_top5 <- F1_Score(test$Class, test$predictedtop5)
F1_stat_top5 


rf_model_top10 <- randomForest(Class ~ V12 + V14 + V11 + V10 + V17 + V3 + V4 + V16 + V6 + V9, data = train)

test$predictedtop10 <- predict(rf_model_top10, test)

F1_stat_top10 <- F1_Score(test$Class, test$predictedtop10)
F1_stat_top10 

#Any number of predictors two or greater seems to yield the same F1 statistic, so we can conclude that
#the model using the two most important factors by Gini impurity is the best one

#Check where increasing the number of trees no longer helps performance for the top two model
plot(rf_model_top2)
plot(rf_model_top2, xlim = c(0,50))

#Performance appears to level off after about 15+ trees

plot(rf_model)
plot(rf_model, xlim = c(0,50))

#For the all-inclusive model, the error is higher and doesnt level off until around 25+ trees
