import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pickle


try:
    crop_data = pd.read_csv("Crop_recommendation.csv")
except FileNotFoundError:
    print("Error: The 'Crop_recommendation.csv' file is missing.")
    exit(1)


unique_crops = crop_data['label'].unique()
crop_dict = {crop: idx for idx, crop in enumerate(unique_crops, start=1)}
reverse_crop_dict = {idx: crop for crop, idx in crop_dict.items()}
crop_data['label'] = crop_data['label'].map(crop_dict)


x = crop_data.drop('label', axis=1)
y = crop_data['label']


feature_ranges = {col: (x[col].min(), x[col].max()) for col in x.columns}


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)


minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

x_train_scaled = minmax_scaler.fit_transform(x_train)
x_test_scaled = minmax_scaler.transform(x_test)
x_train_standardized = standard_scaler.fit_transform(x_train_scaled)
x_test_standardized = standard_scaler.transform(x_test_scaled)


rf_classifier = RandomForestClassifier(random_state=7)
rf_classifier.fit(x_train_standardized, y_train)


y_pred = rf_classifier.predict(x_test_standardized)
print("Model Evaluation Report:")
print(classification_report(y_test, y_pred))


feature_importances = rf_classifier.feature_importances_
feature_names = crop_data.drop('label', axis=1).columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Crop Recommendation')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


with open('model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(minmax_scaler, f)

with open('standardscaler.pkl', 'wb') as f:
    pickle.dump(standard_scaler, f)

with open('crop_dict.pkl', 'wb') as f:
    pickle.dump(crop_dict, f)

with open('reverse_crop_dict.pkl', 'wb') as f:
    pickle.dump(reverse_crop_dict, f)

with open('feature_ranges.pkl', 'wb') as f:
    pickle.dump(feature_ranges, f)

print("Model, scalers, crop dictionaries, and feature ranges have been saved successfully!")




