#########################################
##                                     ##
##    Will it Rain today?              ##
##    PH125.9x  - Capstone Report      ##
##    Christopher Heather              ##
##    15/11/2021                       ##
##                                     ##
#########################################


## ----setup, include=TRUE--------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(warning = FALSE)
knitr::opts_chunk$set(message = FALSE)


## ----Data import----------------------------------------------------------------------------------------------------------------
### Data import

# Load required packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(RANN)) install.packages("RANN", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(lubridate)
library(caret)
library(randomForest)
library(RANN)


# Australian rainfall data https://www.kaggle.com/jsphyg/weather-dataset-rattle-package 
# (accessed 15/11/21)

# rainAUS.csv uploaded to Git repository 
# https://github.com/csheather/harvardX_data_science_capstone_ausRain

# dowload data file from Github repository

dl <- tempfile()
download.file("https://raw.githubusercontent.com/csheather/harvardX_data_science_capstone_ausRain/main/weatherAUS.csv", dl)

# Read file into data frame                   
ausrain <- 
  read_csv(dl) 

# Coerce character columns into factors
ausrain <- ausrain %>%
  mutate(RainToday = as.factor(RainToday),
         RainTomorrow = as.factor(RainTomorrow),
         Location = as.factor(Location),
         WindGustDir = as.factor(WindGustDir),
         WindDir9am = as.factor(WindDir9am),
         WindDir3pm = as.factor(WindDir3pm))

# Clean up
rm(dl)



## ----Data structure,echo=FALSE--------------------------------------------------------------------------------------------------
### Data structure

str(ausrain, vec.len = 2, give.attr=FALSE)



## ----Create Month and Year Columns----------------------------------------------------------------------------------------------
### Create Month and Year Columns

ausrain <- ausrain %>%
  mutate(Month = as.factor(month(Date)),
         Year = as.factor(year(Date))) %>%
  select(-Date)



## ----Identify and remove NA in ausrain$RainTomorrow-----------------------------------------------------------------------------
### Identify and remove NA in ausrain$RainTomorrow

ausrain <- filter(ausrain, !is.na(RainTomorrow))



## ----Creation of validation set-------------------------------------------------------------------------------------------------
### Creation of validation set

# validation set is 10% of initial ausrain set

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = ausrain$RainTomorrow, times = 1, p = 0.1, list = FALSE)
ausrain <- ausrain[-test_index,]
validation <- ausrain[test_index,]
rm(test_index)


## ----Distribution of Location observations--------------------------------------------------------------------------------------
### Distribution of Location observations

ausrain %>% group_by(Location) %>% summarise(count = n()) %>%
  ggplot(aes(reorder(Location, count), count)) +
  geom_col() +
  coord_flip()+
  xlab("")



## ----Distribution of Year observations------------------------------------------------------------------------------------------
### Distribution of Year observations

ausrain %>% group_by(Year) %>% summarise(count = n()) %>%
  ggplot(aes(Year, count)) +
  geom_col()




## ----Significance tests---------------------------------------------------------------------------------------------------------
### Significance tests

# Find column names by type

factor_names <- ausrain %>% select(where(is.factor)) %>% colnames() 
numeric_names <- ausrain %>% select(where(is.numeric)) %>% colnames()


# Perform chi-squared test on categorical parameters

p_values_chisq <- sapply(factor_names, function(x){
  test <- ausrain %>% select(RainTomorrow, x) %>% 
    table() %>% 
    chisq.test()
  return(test$p.value)
})

# Number of non-significant parameters - categorical

sum(p_values_chisq >= 0.05)


# perform T-test on continuous parameters

p_values_T <- sapply(ausrain[,numeric_names], function(x) {
  test <- t.test(x ~ ausrain$RainTomorrow, var.equal = TRUE)
  return(test$p.value)
})

#Number of non-significant parameters - numeric

sum(p_values_T >= 0.05)

#Clean up

rm(factor_names, p_values_chisq, p_values_T)


## ----Find large number of NA----------------------------------------------------------------------------------------------------
### Find large number of NA

too_many_NA <- ausrain %>% summarise(across(.fns = ~mean(is.na(.)) >= 0.3))

colnames(ausrain[,which(too_many_NA == TRUE)])



## ----Remove large number of NA--------------------------------------------------------------------------------------------------
### Remove large number of NA

ausrain <- ausrain %>% select(!which(too_many_NA == TRUE))

rm(too_many_NA)


## ----Near Zero Variance---------------------------------------------------------------------------------------------------------
### Near Zero Variance

nearZeroVar(ausrain)


## ----Highly corrlated variables-------------------------------------------------------------------------------------------------
### Highly corrlated variables

# Names of numeric variables (needs to be updated because some removed)

numeric_names <- ausrain %>% select(where(is.numeric)) %>% colnames()

#Correlation matrix

correlations <- cor(ausrain[,numeric_names], use = "na.or.complete")

#Find highly correlated variables

highlyCorrelated <- findCorrelation(correlations, cutoff=0.9)
colnames(correlations)[highlyCorrelated]



## ----Remove highly correlated variables-----------------------------------------------------------------------------------------
### Remove highly correlated variables

ausrain <- select(ausrain, -colnames(correlations)[highlyCorrelated])

rm(highlyCorrelated, correlations)


## ----Find linear dependencies---------------------------------------------------------------------------------------------------
### Find linear dependencies

# update numeric_names

numeric_names <- ausrain %>% select(where(is.numeric)) %>% colnames()

# Find linear dependencies

linearCombos <- findLinearCombos(na.omit(ausrain[,numeric_names]))

linearCombos

rm(numeric_names, linearCombos)



## -------------------------------------------------------------------------------------------------------------------------------
### Data structure

str(ausrain, vec.len = 2)


## ----Imputing numeric values----------------------------------------------------------------------------------------------------
### Imputing numeric values

# median imputation of numeric values

numimpute_preProcess_object <- preProcess(as.data.frame(ausrain), 
                                method = "medianImpute")

# impute numeric values in ausrain 

ausrain <- predict(numimpute_preProcess_object, newdata = as.data.frame(ausrain))

# use same object to impute numeric values in validation

validation <- predict(numimpute_preProcess_object, newdata = as.data.frame(validation))

# Clean up

rm(numimpute_preProcess_object)


## ----Random Guess---------------------------------------------------------------------------------------------------------------
### Random Guess

#overall likelihood of rain the next day

p_tomorrow <- mean(ausrain$RainTomorrow == "Yes")

#Guess randomly using p_tomorrow

set.seed(1999, sample.kind = "Rounding")
p_hats <- sample(x = c("Yes", "No"), 
                 size = nrow(ausrain), 
                 replace = TRUE, 
                 prob = c(p_tomorrow, 1-p_tomorrow))

# assess accuracy

guess_accuracy <-confusionMatrix(as.factor(p_hats), ausrain$RainTomorrow)$overall[["Accuracy"]]

# Start a table with results

results_table <- tribble(~"Model", ~"Accuracy",
                         "Random Guess", guess_accuracy)

# Clean up

rm(p_tomorrow, p_hats)


## ----Guess on RainToday---------------------------------------------------------------------------------------------------------
### Guess on RainToday

# select RainToday and RainTomorrow, remove NA

df <- ausrain %>% select(RainToday, RainTomorrow) %>% na.omit()

# Assess accuracy

guess_accuracy <- confusionMatrix(df$RainTomorrow, df$RainToday)$overall[["Accuracy"]]

# Add to table

results_table <- add_row(results_table, 
                         Model = "RainToday", 
                         Accuracy = guess_accuracy)

# Clean up

rm(df)


## ----Generalised Linear Model, cache=TRUE---------------------------------------------------------------------------------------
### Generalised Linear Model

# Train logistic regression model. NA are excluded from training data set.

ausrain_glm <- train(RainTomorrow ~., data = ausrain,
                     method = "glm",
                     na.action = na.omit,
                     family = "binomial")

# Generate predictions from validation data using trained model

predictions_glm <- predict(ausrain_glm, 
                           newdata = validation, 
                           type = "raw", 
                           na.action = na.pass)

# Determine accuracy of predictions

glm_accuracy <- confusionMatrix(predictions_glm, validation$RainTomorrow)$overall[["Accuracy"]]

# Add result to the table

results_table <- add_row(results_table, 
                         Model = "Logistic Regression", 
                         Accuracy = glm_accuracy)

rm(guess_accuracy, predictions_glm, ausrain_glm)



## ----Regression Tree,cache=TRUE-------------------------------------------------------------------------------------------------
### Regression Tree

# Train tree

train_rpart <- train(RainTomorrow~., data = ausrain,
      method = "rpart",
      tuneGrid = data.frame(cp = seq(0, 0.05, len = 10)),
      na.action = na.omit)

# Use training set to predict outcomes in validation set

predict_rpart <- predict(train_rpart, 
                         newdata = validation, 
                         na.action = na.pass)

# assess accuracy

rpart_accuracy <- confusionMatrix(predict_rpart,validation$RainTomorrow)$overall[["Accuracy"]]

# Add result to the table
results_table <- add_row(results_table, 
                         Model = "Regression Tree", 
                         Accuracy = rpart_accuracy)


## ----Regression tree hyperparameter---------------------------------------------------------------------------------------------
### Regression tree hyperparameter

plot(train_rpart)


## ----Plotting final tree--------------------------------------------------------------------------------------------------------
### Plotting final tree

plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)

rm(train_rpart, glm_accuracy)


## ----Random Forest,cache=TRUE---------------------------------------------------------------------------------------------------
### Random Forest

# Set number of cross validations

control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = 5) # ~sqrt(p)

# Train using ausrain set

train_rf <-  train(RainTomorrow ~., data = ausrain, 
                   method = "rf", 
                   ntree = 100,
                   trControl = control,
                   tuneGrid = grid,
                   na.action = na.omit)

# Predict using validation set

predict_rf <- predict(train_rf, 
                      newdata = validation,  
                      na.action = na.roughfix)

# Detemine Accuracy

rf_accuracy <- confusionMatrix(predict_rf, validation$RainTomorrow)$overall[["Accuracy"]]

# Add result to the table

results_table <- add_row(results_table, 
                         Model = "Random Forest", 
                         Accuracy = rf_accuracy)

#Clean up

rm(control, train_rf, predict_rf, rpart_accuracy)



## -------------------------------------------------------------------------------------------------------------------------------
### Results table

knitr::kable(results_table)

