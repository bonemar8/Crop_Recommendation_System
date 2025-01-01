from flask import Flask, request, render_template
import pickle
import numpy as np

# Loading the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
standard_scaler = pickle.load(open('standardscaler.pkl', 'rb'))

# Reversing crop dictionary(from integers back to strings)
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans',
    7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
    13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya',
    19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'
}

# Initializing Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting input data from form
        nitrogen = float(request.form['Nitrogen'])
        phosphorus = float(request.form['Phosporus'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Preparing features for prediction
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        features_scaled = minmax_scaler.transform(features)
        features_standardized = standard_scaler.transform(features_scaled)

        # Making prediction
        predicted_label = model.predict(features_standardized)[0]
        recommended_crop = crop_dict[predicted_label]

        return render_template('index.html', result=recommended_crop)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)