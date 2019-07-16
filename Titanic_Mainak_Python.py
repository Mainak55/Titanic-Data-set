# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:44:54 2019

@author: Mainak Sarkar
"""

#Titanic

#Setting the working directory
import os
os.chdir("C:/Users/Mainak Sarkar/Desktop/Kaggle/Titanic")

#Importing the data skipping the blank lines if any
import pandas as pd
data = pd.read_csv("train.csv", skip_blank_lines = True)

#Checking the summary of the whole data set
data.describe(include = "all")

#Checking class of each column
data.dtypes





#***********************************DATA PREPROCESSING***********************************

#Data type conversion
#converting "Survived" as factor
data['Survived'] = data['Survived'].astype('category')

#converting "Pclass" as factor
data['Pclass'] = data['Pclass'].astype('category')

#converting "SibSp" as factor
data['SibSp'] = data['SibSp'].astype('category')

#converting "Parch" as factor
data['Parch'] = data['Parch'].astype('category')



#Removing unnecessary variables
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value = pd.DataFrame(data.isnull().sum())

#Resetting Index
missing_value = missing_value.reset_index()

#Renaming Variable
missing_value = missing_value.rename(columns = {'index':'Variable Name', 0 : 'Missing-Percentage'})

#Calculating Missing Value Percentage
missing_value['Missing-Percentage'] = (missing_value['Missing-Percentage']/len(data))*100

#So we can see that there is large amount of missing value in the "Age" variable
#So we have to go for imputation

#Moreover on going through the dataset precisely we can see that the "Embarked" variable doesn't hold any value for row no. 62 and 830
#And this is not captured as missing value as well
#That means that the rows are containing spaces as the values
#So we have to delete those rows
data = data.drop([61, 829], axis = 0)






#***********************************ONLY FOR FINDING THE BEST METHOD***********************************
#Making a sample to check which method works best
import numpy as np
#Choosing a sample and saving its value
sample_NA = data.loc[49, ['Age']]

#Putting values of sample equal to NA for required columns
data.loc[49, ['Age']] = np.nan


#MEAN Method
data[['Age']] = data[['Age']].fillna(data[['Age']].mean())
    
sample_NA_mean = data.loc[49, ['Age']]


#Re_Run the above part of code without the MEAN Method

#MEDIAN Method
data[['Age']] = data[['Age']].fillna(data[['Age']].median())

sample_NA_median = data.loc[49, ['Age']]

#Comparing different imputing methods
sample = pd.concat([sample_NA, sample_NA_mean, sample_NA_median], axis = 1)

sample.columns = ['sample_NA', 'sample_NA_mean', 'sample_NA_median']
 
#Inserting a new blank row in "sample"
sample['Best Method'] = np.nan

#Finding the best method of imputation for each column
for d in range(sample.shape[0]):
    if  (abs(sample.iloc[d, 0]-sample.iloc[d, 1]) < abs(sample.iloc[d, 0]-sample.iloc[d, 2])):
        sample.iloc[d, 3] = "MEAN"
    else:
        sample.iloc[d, 3] = "MEDIAN"

#From "sample" dataframe we can find the best method for each column





#**************************************************************************************
#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD"
data['Age'] = data['Age'].fillna(data['Age'].median())





#***********************************OUTLIER ANALYSIS***********************************

#Selecting numerical variables
numerical_cnames = ['Age', 'Fare']                       #"instant" and all dependent variables are removed
#passenger_count and fare_amount is dealt separately

import numpy as np

#Detecting Outliers and replacing them with NA's
for a in numerical_cnames:
    q75, q25 = np.percentile(data.loc[:,a], [75, 25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    data.loc[data[a]<min, a] = np.nan
    data.loc[data[a]>max, a] = np.nan


#Checking null values
outliers = pd.DataFrame(data.isnull().sum())

#Resetting Index
outliers = outliers.reset_index()

#Renaming Variables
outliers = outliers.rename(columns = {'index':'Variable Name', 0 : 'Missing-Percentage'})

#Calculating Missing Value Percentage
outliers['Missing-Percentage'] = (outliers['Missing-Percentage']/len(data))*100

#As we know that presence of ouliers can affect our models a lot
#So we either have to delete them or go for imputation
#But as the no. of outliers is high so we will opt for imputation





#***********************************ONLY FOR FINDING THE BEST METHOD***********************************
#Making a sample to check which method works best
#Choosing a sample and saving its value
sample_NA_outliers = data.loc[49, ['Fare', 'Age']]

#Putting values of sample equal to NA for required columns
data.loc[49, ['Fare', 'Age']] = np.nan


#MEAN Method
for b in numerical_cnames :
    data[b] = data[b].fillna(data[b].mean())
    
sample_NA_mean_outliers = data.loc[49, ['Fare', 'Age']]


#Re_Run the above part of code without the MEAN Method

#MEDIAN Method
for c in numerical_cnames :
    data[c] = data[c].fillna(data[c].median())

sample_NA_median_outliers = data.loc[49, ['Fare', 'Age']]

#Comparing different imputing methods
sample_outliers = pd.concat([sample_NA_outliers, sample_NA_mean_outliers, sample_NA_median_outliers], axis = 1)

sample_outliers.columns = ['sample_NA', 'sample_NA_mean', 'sample_NA_median']
 
#Inserting a new blank row in "sample"
sample_outliers['Best Method'] = np.nan

#Finding the best method of imputation for each column
for d in range(sample_outliers.shape[0]):
    if  (abs(sample_outliers.iloc[d, 0]-sample_outliers.iloc[d, 1]) < abs(sample_outliers.iloc[d, 0]-sample_outliers.iloc[d, 2])):
        sample_outliers.iloc[d, 3] = "MEAN"
    else:
        sample_outliers.iloc[d, 3] = "MEDIAN"


#From "sample_outliers" dataframe we can find the best method for each column





#**************************************************************************************
#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD"
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())





#***********************************EXPOLATORY DATA ANALYSIS AND FEATURE SELECTION***********************************

#Correlation analysis
import matplotlib.pyplot as plt
num_cnames = ['Age', 'Fare']
data_corr = data.loc[:, num_cnames]

#Set the height and width of the plot
f,ax = plt.subplots(figsize = (7,5))

#Generate correlation matrix
corr = data_corr.corr()

#plot using seaborn
import seaborn as sns
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool),
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square = True, ax = ax)


#From the correlation plot we can see that there is no high correlation between numerical variables



#Now to check the correlation between the categorical variables we would go for Chi_sq test

cat_names = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

from scipy.stats import chi2_contingency

for j in cat_names:
    print(j)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['Survived'], data[j]))
    print(p)

#The p_value of all the variables are less than 0.05
#So we can say that all the variables are important for our analysis



#Now to check the relationship of numerical variables with the dependent variable we will opt for ANOVA Test

#ANOVA(Analysis of Variance) Test
import statsmodels.api as sm
from statsmodels.formula.api import ols

#ANOVA for Fare
data.boxplot('Fare' , by = 'Survived')
Survived_Fare_ols = ols('Fare ~ Survived', data = data).fit()
sm.stats.anova_lm(Survived_Fare_ols, type = 1)
#So from the plot we can see that the mean of Fare is not same across groups
#As p_value is less than 0.05 so we can say that Fare is an important variable



#ANOVA for Age
data.boxplot('Age' , by = 'Survived')
Survived_Age_ols = ols('Age ~ Survived', data = data).fit()
sm.stats.anova_lm(Survived_Age_ols, type = 1)
#So from the plot we can see that the mean of Age is almost same across groups
#As p_value is greater than 0.05 so we can say that Age is not an important predictor

#Thus we will delete "Age" for further consideration
data = data.drop(["Age"], axis = 1)





#***********************************FEATURE SCALING***********************************

#As there is only one numerical variable so feature scaling is irrelevant





#***********************************TRAIN-TEST SPLIT***********************************
#Checking the number of values in each factor variable
data['Survived'].value_counts()                    #Slightly Biased
data['Pclass'].value_counts()                      #Slightly Biased
data['Sex'].value_counts()                         #Slightly Biased
data['SibSp'].value_counts()                       #Highly biased
data['Parch'].value_counts()                       #Highly biased
data['Embarked'].value_counts()                    #Slightly Biased



#As the "Parch" and "SibSp" variables are heavily biased and some groups have too low number of observations
#So we would convert them into integers
data['SibSp'] = data['SibSp'].astype('int64')
data['Parch'] = data['Parch'].astype('int64')



#So as we can see that the dataset is baised across some categories so if we apply random sampling in this case
#Then there might be a chance that no observations of the low count groups is included
#So we need to apply stratified spliting in this case taking the most correlated variable as reference variable

#From the correlation plot we can see that Embarked would be the best variable to create the strata as it is highly biased and good correlation with dependent variable
np.random.seed(555)

from sklearn.model_selection import train_test_split
#Categorical variable to be set as an array
y = np.array(data['Embarked'])
training_set,test_set = train_test_split(data, test_size = 0.2, stratify = y)





#***********************************ENCODING CATEGORICAL VARIABLES***********************************

#For Training Set
dummies_train = pd.get_dummies(training_set[["Pclass", "Sex", "Embarked"]], drop_first = True)
training_set = pd.concat([dummies_train, training_set[["SibSp", "Parch", "Fare", "Survived"]]], axis = 1)

#For Test Set
dummies_test = pd.get_dummies(test_set[["Pclass", "Sex", "Embarked"]], drop_first = True)
test_set = pd.concat([dummies_test, test_set[["SibSp", "Parch", "Fare", "Survived"]]], axis = 1)





#***********************************MODEL BUILDING***********************************

#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(random_state = 0).fit(training_set.iloc[:, 0:8], training_set['Survived'])
LR_predictions = LR_model.predict(test_set.iloc[:, 0:8])

from sklearn.metrics import confusion_matrix, accuracy_score
cm_LR = confusion_matrix(test_set.iloc[:, 8], LR_predictions)
accuracy_LR = accuracy_score(test_set.iloc[:, 8], LR_predictions)



#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(criterion = 'entropy').fit(training_set.iloc[:, 0:8], training_set['Survived'])
DT_predictions = DT_model.predict(test_set.iloc[:, 0:8])

cm_DT = confusion_matrix(test_set.iloc[:, 8], DT_predictions)
accuracy_DT = accuracy_score(test_set.iloc[:, 8], DT_predictions)



#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100).fit(training_set.iloc[:, 0:8], training_set['Survived'])
RF_predictions = RF_model.predict(test_set.iloc[:, 0:8])

cm_RF = confusion_matrix(test_set.iloc[:, 8], RF_predictions)
accuracy_RF = accuracy_score(test_set.iloc[:, 8], RF_predictions)



#K Nearest Neighbors(KNN)
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 1).fit(training_set.iloc[:, 0:8], training_set['Survived'])
KNN_predictions = KNN_model.predict(test_set.iloc[:, 0:8])

cm_KNN = confusion_matrix(test_set.iloc[:, 8], KNN_predictions)
accuracy_KNN = accuracy_score(test_set.iloc[:, 8], KNN_predictions)



#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB().fit(training_set.iloc[:, 0:8], training_set['Survived'])
NB_predictions = NB_model.predict(test_set.iloc[:, 0:8])

cm_NB = confusion_matrix(test_set.iloc[:, 8], NB_predictions)
accuracy_NB = accuracy_score(test_set.iloc[:, 8], NB_predictions)



#SVM(Support Vector Machine)
from sklearn.svm import SVC
SVC_model = SVC(kernel = 'rbf', random_state = 0).fit(training_set.iloc[:, 0:8], training_set['Survived'])
SVC_predictions = SVC_model.predict(test_set.iloc[:, 0:8])

cm_SVC = confusion_matrix(test_set.iloc[:, 8], SVC_predictions)
accuracy_SVC = accuracy_score(test_set.iloc[:, 8], SVC_predictions)



#The Decision Tree Model gives the best accuracy(= 79%) so we would select it as our Best Model





#*************************************************************************


#TEST DATA


#Importing the data
test1 = pd.read_csv("test.csv", skip_blank_lines = True)



#deleting unnecessary variables
test = test1.drop(["PassengerId", "Name", "Age", "Ticket", "Cabin"], axis = 1)





#***********************************DATA TYPE CONVERSION***********************************

test['Pclass'] = test['Pclass'].astype('category')





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value_output = pd.DataFrame(test.isnull().sum())

#Resetting Index
missing_value_output = missing_value_output.reset_index()

#Renaming Variable
missing_value_output = missing_value_output.rename(columns = {'index':'Variable Name', 0 : 'Missing-Percentage'})

#Calculating Missing Value Percentage
missing_value_output['Missing-Percentage'] = (missing_value_output['Missing-Percentage']/len(test))*100



#So we can see the presence of NA's in input data
#Imputing the missing value by mean method
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())





#***********************************ENCODING CATEGORICAL VARIABLES***********************************

#For Original Data
dummies_data = pd.get_dummies(data[["Pclass", "Sex", "Embarked"]], drop_first = True)
data = pd.concat([dummies_data, data[["SibSp", "Parch", "Fare", "Survived"]]], axis = 1)

#For Test Data
dummies_test_data = pd.get_dummies(test[["Pclass", "Sex", "Embarked"]], drop_first = True)
test = pd.concat([dummies_test_data, test[["SibSp", "Parch", "Fare"]]], axis = 1)





#***********************************MODEL TUNING***********************************

#Now Decision Tree model being the best model we will train it with the whole data set and apply some hyper parameter tuning

from sklearn.tree import DecisionTreeClassifier
Best_model = DecisionTreeClassifier(criterion = 'entropy').fit(data.iloc[:, 0:8], data['Survived'])

parameters = [{'max_features' : ['auto', 'sqrt', 'log2']}]

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = Best_model,
                    param_grid = parameters,
                    cv = 10,
                    n_jobs = - 1)

grid = grid.fit(data.iloc[:, 0:8], data['Survived'])

best_parameters = grid.best_params_

#Training the model with best parameters
Best_Model = grid.best_estimator_.fit(data.iloc[:, 0:8], data['Survived'])





#***********************************PREDICTION OF TEST CASES***********************************
output = pd.DataFrame(Best_Model.predict(test))

output = pd.concat([test1.iloc[:, 0], output], axis = 1)
output.columns = ["PassengerId", "Survived"]



#Saving as csv
output.to_csv(r'C:/Users/Mainak Sarkar/Desktop/Kaggle/Titanic/output_Titanic_Python.csv', index = False)









