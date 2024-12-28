import pandas as pd
import numpy as np

#!!! COMMENTED SECTIONS(beside of this one) are for reviewing the data, they can be removed in later versions

crop_data = pd.read_csv("Crop_recommendation.csv")

# print(crop_data.head()) 

# print(crop_data.head()) 

# print(crop_data.info())

# print(crop_data.isnull().sum()) 

# print(crop_data.duplicated().sum()) 

# print(crop_data.corr())

numeric_data = crop_data.drop(columns=['label'])
# print(numeric_data.corr())

# import seaborn as sns
# sns.heatmap(numeric_data.corr(), annot=True, cbar=True)  #this doesn't work in vs code, need google colab or jupyter

# print(crop_data.label.value_counts())

# print(crop_data['label'].unique()) #to see what crops we have

#LABEL ENCODING

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

#ABOVE WE TURNED LABELS INTO int's because machine learning algorithm sometimes do not work with strings, like corr() function

# print(crop_data.head()) #Now labels are numbers as you can notice
