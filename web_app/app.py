from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)

FAKE_NEWS_MODEL_DIR = os.path.join(PROJECT_ROOT, 'fake_news_detector', 'saved_model')
DEEPFAKE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'deepfake_detector', 'saved_model')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

fn_model = None
fn_vectorizer = None
df_model = None

try:
    with open(os.path.join(FAKE_NEWS_MODEL_DIR, 'fn_model.pkl'), 'rb') as model_file:
        fn_model = pickle.load(model_file)
    with open(os.path.join(FAKE_NEWS_MODEL_DIR, 'fn_vectorizer.pkl'), 'rb') as vectorizer_file:
        fn_vectorizer = pickle.load(vectorizer_file)
    print("Fake news model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Fake news model or vectorizer not found. Please run the training script.")

try:
    DF_MODEL_PATH = os.path.join(DEEPFAKE_MODEL_DIR, 'df_model.h5')
    df_model = load_model(DF_MODEL_PATH)
    print(f"Deepfake model loaded successfully from {DF_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load the deepfake model. Video analysis will not work.")
    print(f"Error details: {e}")

def predict_deepfake(video_path):
    if df_model is None:
        return "Error: Deepfake model not loaded"

    IMG_SIZE = 128
    FRAME_SAMPLE_RATE = 15
    try:
        camera = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            if frame_count % FRAME_SAMPLE_RATE == 0:
                resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                normalized_frame = resized_frame / 255.0
                reshaped_frame = np.expand_dims(normalized_frame, axis=0)
                prediction = df_model.predict(reshaped_frame, verbose=0)[0][0]
                
                if prediction < 0.5:
                    camera.release()
                    return 'FAKE'
            
            frame_count += 1
            
        camera.release()
        return 'REAL'

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return "Error during analysis"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_news', methods=['POST'])
def predict_news():
    if request.method == 'POST' and fn_model and fn_vectorizer:
        news_text = request.form['news_text']
        vectorized_text = fn_vectorizer.transform([news_text])
        prediction = fn_model.predict(vectorized_text)[0]
        result_class = 'result-real' if prediction == 'REAL' else 'result-fake'
        return render_template('new_results.html', prediction=prediction, class_name=result_class)
    return redirect(url_for('home'))

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['video_file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        prediction = predict_deepfake(video_path)
        
        os.remove(video_path)
        
        result_class = 'result-real' if prediction == 'REAL' else 'result-fake'
        return render_template('video_results.html', prediction=prediction, class_name=result_class)

    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5000)