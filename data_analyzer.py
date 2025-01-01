import pandas as pd
import numpy as np

#!!! COMMENTED SECTIONS(beside of this one) are for reviewing the data, they can be removed in later versions
#Started to explain what some code does in comments, you might not wanna delete them till latest version(your choice)

crop_data = pd.read_csv("Crop_recommendation.csv")

# print(crop_data.head()) 

# print(crop_data.head()) 

# print(crop_data.info())

# print(crop_data.isnull().sum()) 

# print(crop_data.duplicated().sum()) 

# print(crop_data.corr())

# numeric_data = crop_data.drop(columns=['label']) #for corr function, wasn't working with strings
# print(numeric_data.corr())

# import seaborn as sns
# sns.heatmap(numeric_data.corr(), annot=True, cbar=True)  #this doesn't work in vs code, need google colab or jupyter

# print(crop_data.label.value_counts())

# print(crop_data['label'].unique()) #to see what crops we have

#LABEL ENCODING:

crop_dict = {
  'rice' : 1,
  'maize' : 2,
  'chickpea' : 3,
  'kidneybeans' : 4,
  'pigeonpeas' : 5,
  'mothbeans' : 6,
  'mungbean' : 7,
  'blackgram' : 8,
  'lentil' : 9,
  'pomegranate' : 10,
  'banana' : 11,
  'mango' : 12,
  'grapes' : 13,
  'watermelon' : 14,
  'muskmelon' : 15,
  'apple' : 16,
  'orange' : 17,
  'papaya' : 18,
  'coconut' : 19,
  'cotton' : 20,
  'jute' : 21,
  'coffee' : 22,
}  #by crop_data.label.value_counts() order

crop_data['label'] = crop_data['label'].map(crop_dict) #maps labels(keys in dict) to their values (integers)

#ABOVE WE TURNED LABELS INTO integers because machine learning algorithm sometimes do not work with strings, like corr() function

# print(crop_data.head()) #Now labels are numbers as you can notice

# print(crop_data.label.value_counts()) #100 of each label but now as integers from 1 to 22 (22 labels)

#FEATURE AND TARGET SEPERATION: (seperating labels and corresponding values to different variables)

x = crop_data.drop('label', axis = 1)
y = crop_data['label']

# print(x.head())
# print(y.head())

#SPLITTING DATA FOR TRAINING AND TESTING:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)

#%30 is used for resting %70 is used for trainin the data 
#80-20 ratio is what generally used but i want better evaluation (20 felt low, might be irrational)
#random_state's value doesn't matter, BUT it needs to be consistent (if 7, 7 everywhere)

# print(x_test.shape)
# print(x_train.shape)
#by rows you can see the split percentage, change test_size and try above again to see difference

#SCALING THE DATA:

from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()

x_train = mx.fit_transform(x_train)
x_test = mx.transform(x_test)

#The point is to scale to data between 0-1 so they perform better in some algorithms,
#We took min and max values for each feature and map 0 to Xmin, 1 to Xmax

# print(x_train) #Now you can see each value is between 0 and 1

#STANDARDIZING DATA:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(x_train)  #Calculates mean and standard deviation for each feature in x_train
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#this centers the data (mean = 0) and scale it (standard deviation = 1).

#TRAINING DATA WITH DIFFERENT CLASSIFIERS:

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

models = {
  'LogisticRegression' : LogisticRegression(),
  'GaussianNB' : GaussianNB(),
  'SVC' : SVC(),
  'KNeighborsClassifier' : KNeighborsClassifier(),
  'DecisionTreeClassifier' : DecisionTreeClassifier(),
  'ExtraTreeClassifier' : ExtraTreeClassifier(),
  'RandomForestClassifier' : RandomForestClassifier(),
  'BaggingClassifier' : BaggingClassifier(),
  'GradientBoostingClassifier' : GradientBoostingClassifier(),
  'AdaBoostClassifier' : AdaBoostClassifier(),
}

# for name, model in models.items():
#   model.fit(x_train, y_train)
#   y_prediction = model.predict(x_test)
#   score = accuracy_score(y_test, y_prediction) 
#   print(f"{name} model's accuracy is: {score}")
# Testing each classifier to see which one has better prediction rate
# accuracy_score() function evaluates performance of the model by correct/total number of predictions
# 1. accuracy is for RandomForestClassifier, 2. was GaussianNB model

rc = RandomForestClassifier()
rc.fit(x_train, y_train)
y_prediction = rc.predict(x_test)
# print(accuracy_score(y_test, y_prediction))

#TESTING RECOMMENDATION:

# print(crop_data.columns)
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
  features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

  mx_features = mx.fit_transform(features)
  sc_mx_features = sc.transform(mx_features)

  prediction = rc.predict(sc_mx_features).reshape(1, -1)
  return prediction[0]

# print(crop_data.head(10))

#Test values(the first row of crop_data):
N = 90
P = 42
K = 43
temperature = 20.879744
humidity = 82.002744
ph = 6.502985
rainfall = 202.935536

#The recommendation for given values
predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)

print(f"Recommended crop is: {predicted_crop}") #will give you a number since we mapped crops to integers earlier


