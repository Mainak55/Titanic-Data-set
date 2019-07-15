#Titanic

#Clearing the workspace
rm(list = ls())

#Setting the working directory
setwd("C:/Users/Mainak Sarkar/Desktop/Kaggle/Titanic")

#Importing the data skipping the blank lines if any
data = read.csv("train.csv", blank.lines.skip = TRUE)

#Checking the summary of the whole data set
summary(data)

#Checking class of each column
class(data$PassengerId)                            #integer
class(data$Survived)                               #integer
class(data$Pclass)                                 #integer
class(data$Name)                                   #factor
class(data$Sex)                                    #factor
class(data$Age)                                    #numeric
class(data$SibSp)                                  #integer
class(data$Parch)                                  #integer
class(data$Ticket)                                 #factor
class(data$Fare)                                   #numeric
class(data$Cabin)                                  #factor
class(data$Embarked)                               #factor





#***********************************DATA PREPROCESSING***********************************

#Data type conversion

#converting "Survived" as factor
data$Survived = as.factor(data$Survived)

#converting "Pclass" as factor
data$Pclass = as.factor(data$Pclass)

#converting "Sibsp" as factor
data$SibSp = as.factor(data$SibSp)

#converting "Parch" as factor
data$Parch = as.factor(data$Parch)



#Removing unnecessary variables
data = data[, -c(1, 4, 9, 11)]





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value = data.frame(apply(data, 2, function(x){sum(is.na(x))}))

#Calculating Percentage of missing values
missing_value$percentage = (missing_value[, 1]/nrow(data))*100

#So we can see that there is large amount of missing value in the "Age" variable
#So we have to go for imputation

#Moreover on going through the dataset precisely we can see that the "Embarked" variable doesn't hold any value for row no. 62 and 830
#And this is not captured as missing value as well
#That means that the rows are containing spaces as the values
#So we have to delete those rows
data = data[ -c(62, 830), ]



#***********************************ONLY FOR FINDING THE BEST METHOD***********************************

#Choosing the best method for missing value imputation

#Making a sample to check which method works best
#Choosing a sample and saving its value
sample_NA = data[50, 4]

#Putting values of sample equal to NA for required columns
data[50, 4] = NA

#duplicating data
data_duplicate = data

#MEAN Method
data[, 4][is.na(data[, 4])] = mean(data[, 4], na.rm = TRUE)

sample_NA_mean = data[50, 4]

#MEDIAN Method
data = data_duplicate
data[, 4][is.na(data[, 4])] = median(data[, 4], na.rm = TRUE)

sample_NA_median = data[50, 4]

#Comparing different imputing methods
sample = as.data.frame(rbind(sample_NA, sample_NA_mean, sample_NA_median))

#Inserting a new blank row in "sample"
sample[nrow(sample)+1, ]=NA

#Changing row names
row.names(sample) = c("sample_NA","sample_NA_mean","sample_NA_median","Best Method")

colnames(sample) = c("Age")

#Finding the best method of imputation for each column
for (d in (1:ncol(sample)))
{
  if(abs(as.numeric(sample[1,d])-as.numeric(sample[2,d]))<abs(as.numeric(sample[1,d])-as.numeric(sample[3,d])))
  {
    sample[4,d] = "MEAN"
  } else {
    sample[4,d] = "MEDIAN"
  }
}

#From "sample" dataframe we can find the best method for each column





#**************************************************************************************

#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD"
data$Age[is.na(data$Age)] = median(data$Age, na.rm = TRUE)





#***********************************OUTLIER ANALYSIS***********************************

#Selecting numerical variables
numeric_index = sapply(data, is.numeric)
numeric_data = data[, numeric_index]
numerical_cnames = colnames(numeric_data)

#Imputing null values in place of outliers
for (i in numerical_cnames[])                    #"instant" and all dependent variables are removed
{
  val = data[,i][data[,i]%in%boxplot.stats(data[,i])$out]
  data[,i][data[,i]%in%val] = NA
}

#Checking number of outliers(null values)
outliers = data.frame(apply(data, 2, function(y){sum(is.na(y))}))

#Calculating Percentage of outliers(null values)
outliers$percentage = (outliers[,1]/nrow(data))*100

#As we know that presence of ouliers can affect our models a lot
#So we either have to delete them or go for imputation
#But as the no. of outliers is high so we will opt for imputation





#***********************************ONLY FOR FINDING THE BEST METHOD***********************************

#Selecting variables containing NAs
NA_index = sapply(data, anyNA)
NA_data = data[, NA_index]
NA_cnames = colnames(NA_data)

#Choosing the best method for missing value imputation

#Making a sample to check which method works best
#Choosing a sample and saving its value
sample_NA_outliers = data[50, c(4,7)]

#Putting values of sample equal to NA for required columns
data[50,c(NA_cnames)] = NA

#duplicating data
data_duplicate_outliers = data

#MEAN Method
for(b in NA_cnames)
  data[, b][is.na(data[, b])] = mean(data[, b], na.rm = TRUE)

sample_NA_mean_outliers = data[50, c(4,7)]

#MEDIAN Method
data = data_duplicate_outliers
for(c in NA_cnames)
  data[, c][is.na(data[, c])] = median(data[, c], na.rm = TRUE)

sample_NA_median_outliers = data[50, c(4,7)]

#Comparing different imputing methods
sample_outliers = rbind(sample_NA_outliers, sample_NA_mean_outliers, sample_NA_median_outliers)

#Inserting a new blank row in "sample"
sample_outliers[nrow(sample_outliers)+1, ]=NA

#Changing row names
row.names(sample_outliers) = c("sample_NA","sample_NA_mean","sample_NA_median","Best Method")

#Finding the best method of imputation for each column
for (d in (1:ncol(sample_outliers)))
{
  if(abs(as.numeric(sample_outliers[1,d])-as.numeric(sample_outliers[2,d]))<abs(as.numeric(sample_outliers[1,d])-as.numeric(sample_outliers[3,d])))
  {
    sample_outliers[4,d] = "MEAN"
  } else {
    sample_outliers[4,d] = "MEDIAN"
  }
}

#From "sample" dataframe we can find the best method for each column





#**************************************************************************************

#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD" after missing value imputation
data$Age[is.na(data$Age)] = median(data$Age, na.rm = TRUE)
data$Fare[is.na(data$Fare)] = mean(data$Fare, na.rm = TRUE)


#Setting the dependent variable as the last column
data = data[, c(2,3,4,5,6,7,8,1)]





#***********************************EXPOLATORY DATA ANALYSIS AND FEATURE SELECTION***********************************

#Correlation Analysis
cor_index = sapply(data, is.numeric)

library(corrgram)
corrgram(data[, cor_index], order = FALSE, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

#From the correlation plot we can see that there is no high correlation between numerical variables



#Now to check the correlation between the categorical variables we would go for Chi_sq test

factor_index = sapply(data, is.factor)
factor_data = data[, factor_index]

for(k in (1:ncol(factor_data)))
{
  print(names(factor_data)[k])
  print(chisq.test(table(factor_data$Survived, factor_data[, k])))
}

#The p_value of all the variables are less than 0.05
#So we can say that all the variables are important for our analysis



#Now to check the relationship of numerical variables with the dependent variable we will opt for ANOVA Test

#ANOVA(Analysis of Variance) Test

#ANOVA for Fare
plot(Fare ~ Survived, data = data)
#So from the plot we can see that the mean of Fare is not same across groups

#Checking the p_value
summary(aov(Fare ~ Survived, data = data))
#As p_value is less than 0.05 so we can say that Fare is an important variable



#ANOVA for Age
plot(Age ~ Survived, data = data)
#So from the plot we can see that the mean of Age is almost same across groups

#Checking the p_value
summary(aov(Age ~ Survived, data = data))
#As p_value is greater than 0.05 so we can say that Age is not an important predictor

#Thus we will delete "Age" for further consideration
data = data[, -3]



#Renaming the categories
library(plyr)
data$Embarked = revalue(data$Embarked, c("C" = "1", "S" = "2", "Q" = "3"))
data$Sex = revalue(data$Sex, c("female" = "1", "male" = "2"))





#***********************************FEATURE SCALING***********************************

#As there is only one numerical variable so feature scaling is irrelevant





#***********************************TRAIN-TEST SPLIT***********************************

table(data$Pclass)                     #Slightly Biased
table(data$Sex)                        #Slightly Biased
table(data$SibSp)                      #Heavily Biased
table(data$Parch)                      #Heavily Biased
table(data$Embarked)                   #Slightly Biased
table(data$Survived)                   #Slightly Biased



#As the "Parch" and "SibSp" variables are heavily biased and some groups have too low number of observations
#So we would convert them into integers
data$Parch = as.numeric(as.character(data$Parch))
data$SibSp = as.numeric(as.character(data$SibSp))

#Empty records are renamed as "missing"
levels(data$Embarked)[1] = "missing"



#So as we can see that the dataset is baised across some categories so if we apply random sampling in this case
#Then there might be a chance that no observations of the low count groups is included
#So we need to apply stratified spliting in this case taking the most correlated variable as reference variable

#From the correlation plot we can see that Embarked would be the best variable to create the strata as it is highly biased and good correlation with dependent variable
set.seed(123)
library(caret)
train.index = createDataPartition(data$Embarked, p = 0.8, list = FALSE)
training_set = data[train.index,]
test_set = data[-train.index,]





#***********************************MODEL BUILDING***********************************

#Logistic Regression
LR_model = glm(Survived ~ ., data = training_set, family = "binomial")
summary(LR_model)
Logit_predictions = predict(LR_model, newdata = test_set[, -7], type = "response")
LR_predictions = ifelse(Logit_predictions > 0.5, 1, 0)

conf_matrix_LR = table(test_set$Survived, LR_predictions)
confusionMatrix(conf_matrix_LR)



#Decision Tree Classifier
library(C50)
C50_model = C5.0(Survived ~ ., training_set, trials = 100, rules = TRUE)
C50_predictions = predict(C50_model, test_set[, -7], type = "class")

conf_matrix_C50 = table(test_set$Survived, C50_predictions)
confusionMatrix(conf_matrix_C50)



#Random Forest
library(randomForest)
RF_model = randomForest(Survived ~ ., training_set, ntree = 500)
RF_predictions = predict(RF_model, test_set[, -7], type = "class")

conf_matrix_RF = table(test_set$Survived, RF_predictions)
confusionMatrix(conf_matrix_RF)



#K Nearest Neighbors(KNN)
library(class)
KNN_predictions = knn(training_set[, 1:6], test_set[, 1:6], training_set$Survived, k = 1)

conf_matrix_KNN = table(test_set$Survived, KNN_predictions)
confusionMatrix(conf_matrix_KNN)

#As the dependent variable is categorical so we have imputed only the odd values ok
#K = 1 gives the best value



#Naive Bayes
library(e1071)
NB_model = naiveBayes(Survived ~ ., data = training_set)
NB_predictions = predict(NB_model, test_set[, 1:6], type = "class")

conf_matrix_NB = table(test_set$Survived, NB_predictions)
confusionMatrix(conf_matrix_NB)



#SVM(Support Vector Machine)
library(e1071)
SVM_model = svm(formula = Survived ~ .,
                data = training_set,
                type = "C-classification",
                kernel = "linear")
SVM_predictions = predict(SVM_model, type = "response", newdata = test_set[, -7])

conf_matrix_SVM = table(test_set$Survived, SVM_predictions)
confusionMatrix(conf_matrix_SVM)



#ANN
library(h2o)
h2o.init(nthreads = -1)
ANN_model = h2o.deeplearning(y = "Survived",
                             training_frame = as.h2o(training_set),
                             activation = "Rectifier",
                             hidden = c(4,4),
                             epochs = 100,
                             train_samples_per_iteration = -2)
prod_pred = as.data.frame(h2o.predict(ANN_model, newdata = as.h2o(test_set[, -7])))
ANN_predictions = prod_pred[, 1]

conf_matrix_ANN = table(test_set$Survived, ANN_predictions)
confusionMatrix(conf_matrix_ANN)

h2o.shutdown()
Y



#The Random Forest Model gives the best accuracy(= 79%) so we would select it as our Best Model





#*************************************************************************


#TEST DATA


#Importing the data
test1 = read.csv("test.csv")


#deleting unnecessary variables
test = test1[, -c(1, 3, 5, 8, 10)]





#***********************************DATA TYPE CONVERSION***********************************

test$Pclass = as.factor(test$Pclass)



#Renaming the categories
test$Embarked = revalue(test$Embarked, c("C" = "1", "S" = "2", "Q" = "3"))
test$Sex = revalue(test$Sex, c("female" = "1", "male" = "2"))





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value_output = data.frame(apply(test, 2, function(x){sum(is.na(x))}))

#Calculating Percentage of missing values
missing_value_output$percentage = (missing_value_output[, 1]/nrow(test))*100



#So we can see the presence of NA's in input data
#Imputing the missing value by mean method

test$Fare[is.na(test$Fare)] = mean(test$Fare, na.rm = TRUE)





#***********************************MODEL TUNING***********************************

#Now Random Forest model being the best model we will train it with the whole data set and apply some hyper parameter tuning


library(caret)
set.seed(123)
Best_model = train(form = Survived ~ ., data = data[], 
                    method = 'rf', metric = 'Accuracy')
Best_model





#***********************************PREDICTION OF TEST CASES***********************************

output = as.data.frame(predict(Best_model , newdata = test[], type = "raw"))

output = as.data.frame(cbind(test1$PassengerId, output))

colnames(output) = c("PassengerId", "Survived")

#Saving as csv
write.csv(output, "output_Titanic_R.csv", row.names = FALSE)




