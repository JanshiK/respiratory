from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def welcome():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    conn.close()

    if 'user' in session:
        return redirect(url_for('index'))
    elif user_count == 0:
        return redirect(url_for('register'))
    else:
        return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    prediction = ""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'audio.wav'
            file.save(file_path)
            prediction = parkavi(file_path)
    return render_template('index.html', prediction=prediction)

def parkavi(mp):
    def extract_features(file_name):
        max_pad_len = 862
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        except Exception as e:
            print("Error encountered while parsing file:", file_name)
            return None
        return mfccs

    @tf.autograph.experimental.do_not_convert
    def predict(mp):
        D_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
        data = extract_features(mp)
        if data is None:
            return "Feature extraction failed."
        features = np.array([data])
        model = tf.keras.models.load_model('resp_model_300.h5')
        result = model.predict(features, verbose=0)
        dp = list(zip(D_names, list(*result)))
        res = max(dp, key=lambda x: x[1])
        return f"{res[0]}: {res[1]:.2%}"
    return predict(mp)

if __name__ == '__main__':
    app.run(debug=True)
