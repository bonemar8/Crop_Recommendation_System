import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
crop_data = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels into integers
crop_dict = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5, 'mothbeans': 6, 
    'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12, 
    'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16, 'orange': 17, 'papaya': 18, 
    'coconut': 19, 'cotton': 20, 'jute': 21, 'coffee': 22
}

crop_data['label'] = crop_data['label'].map(crop_dict)

# Separate features and target
x = crop_data.drop('label', axis=1)
y = crop_data['label']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# Scale and standardize data
mx = MinMaxScaler()
sc = StandardScaler()

x_train = mx.fit_transform(x_train)
x_test = mx.transform(x_test)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train RandomForest model
rc = RandomForestClassifier()
rc.fit(x_train, y_train)

# Test accuracy
y_prediction = rc.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)
print(f"RandomForestClassifier accuracy: {accuracy}")

# Recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = mx.transform(features)
    features_standardized = sc.transform(features_scaled)
    prediction = rc.predict(features_standardized)
    return prediction[0]

# Test the recommendation function
N, P, K = 90, 42, 43
temperature, humidity, ph, rainfall = 20.879744, 82.002744, 6.502985, 202.935536
predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)

# Output recommended crop
inverse_crop_dict = {v: k for k, v in crop_dict.items()}
print(f"Recommended crop: {inverse_crop_dict[predicted_crop]}")

# Save models and scalers
pickle.dump(rc, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standardscaler.pkl', 'wb'))

#NOTE: if you want to rest recommendation function, comment pickles to not create the files again