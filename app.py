from flask import Flask, request, render_template

from flask_wtf import FlaskForm
from wtforms import FloatField
from wtforms.validators import InputRequired, NumberRange

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

# Flask form class
class CropForm(FlaskForm):
    nitrogen = FloatField('Nitrogen', validators=[
        InputRequired(message="Nitrogen is required."),
        NumberRange(min=0, max=200, message="Nitrogen must be between 0 and 200.")
    ])
    phosphorus = FloatField('Phosphorus', validators=[
        InputRequired(message="Phosphorus is required."),
        NumberRange(min=0, max=200, message="Phosphorus must be between 0 and 200.")
    ])
    potassium = FloatField('Potassium', validators=[
        InputRequired(message="Potassium is required."),
        NumberRange(min=0, max=200, message="Potassium must be between 0 and 200.")
    ])
    temperature = FloatField('Temperature (°C)', validators=[
        InputRequired(message="Temperature is required."),
        NumberRange(min=0, max=50, message="Temperature must be between 0 and 50°C.")
    ])
    humidity = FloatField('Humidity (%)', validators=[
        InputRequired(message="Humidity is required."),
        NumberRange(min=0, max=100, message="Humidity must be between 0 and 100.")
    ])
    ph = FloatField('pH', validators=[
        InputRequired(message="pH is required."),
        NumberRange(min=0, max=14, message="pH must be between 0 and 14.")
    ])
    rainfall = FloatField('Rainfall (mm)', validators=[
        InputRequired(message="Rainfall is required."),
        NumberRange(min=0, max=5000, message="Rainfall must be between 0 and 5000 mm.")
    ])

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'asdqwe123'  # Required for form validation

@app.route('/', methods=['GET', 'POST'])
def home():
    form = CropForm()
    if form.validate_on_submit():
        nitrogen = form.nitrogen.data
        phosphorus = form.phosphorus.data
        potassium = form.potassium.data
        temperature = form.temperature.data
        humidity = form.humidity.data
        ph = form.ph.data
        rainfall = form.rainfall.data

        # Preparing features for prediction
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        features_scaled = minmax_scaler.transform(features)
        features_standardized = standard_scaler.transform(features_scaled)

        # Making prediction
        predicted_label = model.predict(features_standardized)[0]
        recommended_crop = crop_dict[predicted_label]

        return render_template('index.html', result=f"Recommended Crop: {recommended_crop}", form=form)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)