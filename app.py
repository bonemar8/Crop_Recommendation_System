from flask import Flask, render_template, redirect, url_for, flash, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, SubmitField
from wtforms.validators import InputRequired, Email, EqualTo, Length, NumberRange
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load model and necessary files
try:
    model = pickle.load(open('model.pkl', 'rb'))
    minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
    standard_scaler = pickle.load(open('standardscaler.pkl', 'rb'))
    crop_dict = pickle.load(open('crop_dict.pkl', 'rb'))
    reverse_crop_dict = pickle.load(open('reverse_crop_dict.pkl', 'rb'))
    feature_ranges = pickle.load(open('feature_ranges.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class SoilRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    record_name = db.Column(db.String(100), nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[InputRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

class SoilForm(FlaskForm):
    record_name = StringField('Record Name', validators=[InputRequired()])
    nitrogen = FloatField('Nitrogen', validators=[InputRequired(), NumberRange(min=0, max=200)])
    phosphorus = FloatField('Phosphorus', validators=[InputRequired(), NumberRange(min=0, max=200)])
    potassium = FloatField('Potassium', validators=[InputRequired(), NumberRange(min=0, max=200)])
    temperature = FloatField('Temperature (°C)', validators=[InputRequired(), NumberRange(min=0, max=50)])
    humidity = FloatField('Humidity (%)', validators=[InputRequired(), NumberRange(min=0, max=100)])
    ph = FloatField('pH', validators=[InputRequired(), NumberRange(min=0, max=14)])
    rainfall = FloatField('Rainfall (mm)', validators=[InputRequired(), NumberRange(min=0, max=5000)])
    submit = SubmitField('Recommend Crop')

# Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    form = SoilForm()
    if form.validate_on_submit():
        record_name = form.record_name.data
        nitrogen = form.nitrogen.data
        phosphorus = form.phosphorus.data
        potassium = form.potassium.data
        temperature = form.temperature.data
        humidity = form.humidity.data
        ph = form.ph.data
        rainfall = form.rainfall.data

        warnings = []
        for feature, value in zip(feature_ranges.keys(), [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
            min_val, max_val = feature_ranges[feature]
            if not (min_val <= value <= max_val):
                warnings.append(f"{feature.capitalize()} ({value}) is out of range ({min_val}-{max_val})")

        if warnings:
            flash(", ".join(warnings), 'warning')

        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        scaled_features = minmax_scaler.transform(input_features)
        standardized_features = standard_scaler.transform(scaled_features)
        predicted_label = model.predict(standardized_features)[0]
        recommended_crop = reverse_crop_dict.get(predicted_label, "Unknown Crop")

        if recommended_crop != "Unknown Crop":
            new_record = SoilRecord(
                user_id=current_user.id,
                record_name=record_name,
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall
            )
            db.session.add(new_record)
            db.session.commit()

            return render_template(
                'index.html',
                form=form,
                result=f"Recommended Crop: {recommended_crop}",
                records=SoilRecord.query.filter_by(user_id=current_user.id).all()
            )
        else:
            flash("Couldn't recommend a crop. Please check your input values.", 'danger')

    return render_template('index.html', form=form, records=SoilRecord.query.filter_by(user_id=current_user.id).all())

@app.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    record = SoilRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        abort(403)
    return render_template('record_details.html', record=record)

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare_records():
    records = SoilRecord.query.filter_by(user_id=current_user.id).all()
    
    if request.method == 'POST':
        selected_record_ids = request.form.getlist('record_ids')
        selected_records = SoilRecord.query.filter(SoilRecord.id.in_(selected_record_ids)).all()
        return render_template('compare.html', records=selected_records, selected=True)
    
    return render_template('compare.html', records=records, selected=False)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)



