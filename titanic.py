#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats
import pandas as pd
import numpy as np
import sklearn
import warnings
import sys
import os

DeprecationWarning('ignore')
os.chdir('C:/Users/_yuv_/Desktop/mac_learn/data')
warnings.filterwarnings('ignore')


#%%
df=pd.read_csv('train.csv')   # importing the train file which is a historical file
#%% To check for the head, tail and sample
df.head()

df.tail()

df.sample()


#%% To check for the statistical information

df.describe()

#%% To check for the columns information 
df.info()

#%% To check for the columns name 
df.columns

#%% to Divide the data into train and test
from sklearn.model_selection import train_test_split
train, test= train_test_split(df, test_size=0.2, random_state=12)
#  deleting the df variable to save ram
#del df

#%% analysis on train
train.isnull().sum()


#%%   Filling the null data
def fill_age(df):  # filling age with mean 
    mean=29.67
    df['Age'].fillna(mean, inplace=True)
    return df

def fill_Embarked(df):   # Filling embarked with mode
    df.Embarked.fillna('S', inplace=True)
    return df

def dropping_feature(df): # droppinf invalid features
    df=df.drop(['PassengerId','Cabin','Name','Ticket'], axis=1)
    return df

def label_encode(df):   # chaning the labels from string to numbers of the following columns
    from sklearn.preprocessing import LabelEncoder
    label=LabelEncoder()
    df['Sex']=label.fit_transform(df['Sex'])
    df['Embarked']=label.fit_transform(df['Embarked'])
    return df

def encode_feature(df):   # merging all the functions in 1 functions, master function 
    df=fill_age(df)
    df=fill_Embarked(df)
    df= dropping_feature(df)
    df= label_encode(df)
    return df


train=encode_feature(train)  # applying to train 
test= encode_feature(test)   #applying to test
#%%   Dropping the unwanted variables from x and seprating the y variable
def x_and_y(df):
    x = df.drop(["Survived"], axis=1)
    y = df['Survived']
    return x, y

x_train, y_train = x_and_y(train)  # applying upper function on train to create x_train and y_train
x_test, y_test= x_and_y(test)      # applying upper function on test to create x_test and y_test



''' We don't want now train file becasuse it is converted into x_train and y_train,
same with test file x_test, y_test'''

#%%  removing train and test
del train, test

#%% making logistic model
log_model= LogisticRegression()
log_model.fit(x_train, y_train)   # making a model to learn from data and its labels
prediction =log_model.predict(x_test)  # making log model to predict on unseen data
test_score = accuracy_score(y_test, prediction)  # this score is on unseen data
train_score= accuracy_score( y_train, log_model.predict(x_train))  
''' this is to check score on train data how much model learns from data and 
then making prediction on test data'''      
print(f"This is train  score {train_score}")
print(f"\nThis is test score {test_score}")


#%% If you like the score of model then you will make prediction on future data
future_data= pd.read_csv('test.csv')

#%% 
'''all the things that you applied on train data should be applied on test data or 
future data as well '''
future_data= encode_feature(future_data)

# checking of the future_data is null or not
future_data.isnull().sum()  # Fare is null that's why u should not apply the prediction

# 
def fill_fare(df):
    df.Fare.fillna(df.Fare.median(), inplace= True)
    return df

future_data= fill_fare(future_data)

future_prediction= log_model.predict(future_data)


#%% To see the future_prediction 
future_prediction

#%% To find out the probability
proba= log_model.predict_proba(future_data)  # 0th position is of serial number can be ignored, while 1st position is the probaility
proba


#%%

