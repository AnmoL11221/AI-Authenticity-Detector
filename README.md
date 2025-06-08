
# AI Authenticity Detector

A web-based application designed to combat digital misinformation by integrating two distinct AI models: a text-based fake news detector and a video-based deepfake analyzer.


*(Note: You can replace the URL above with a link to a screenshot of your own running application.)*

---

## Features

-   **Dual-Functionality Interface:** A clean, tabbed interface to switch between the Fake News Detector and the Deepfake Detector.
-   **Text-Based Fake News Analysis:** Leverages Natural Language Processing (NLP) with a `PassiveAggressiveClassifier` to analyze the linguistic patterns of news articles and classify them as "REAL" or "FAKE".
-   **Video-Based Deepfake Analysis:** Uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to analyze video files frame-by-frame, identifying visual artifacts characteristic of deepfakes.
-   **User-Friendly Feedback:** Provides clear, immediate feedback on the analysis results and includes a loading state to inform the user that processing is underway.

---

## Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning / Deep Learning:** Scikit-learn, TensorFlow, Keras
-   **Data Handling & Processing:** Pandas, NumPy, OpenCV
-   **Frontend:** HTML, CSS, JavaScript

---

## Project Structure

```
ai-authenticity-detector/
│
├── fake_news_detector/
│ ├── dataset/ (Contains True.csv & Fake.csv)
│ ├── saved_model/ (Contains fn_model.pkl & fn_vectorizer.pkl)
│ └── train_news_model.py
│
├── deepfake_detector/
│ ├── dataset/ (Used for training on Kaggle)
│ ├── saved_model/ (Contains the downloaded df_model.h5)
│ └── (Training scripts for reference)
│
├── web_app/
│ ├── static/css/style.css
│ ├── templates/
│ │ ├── index.html
│ │ ├── new_results.html
│ │ └── video_results.html
│ ├── uploads/ (Temporary storage for video uploads)
│ └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### 1. Prerequisites
-   Python 3.8+
-   `pip` (Python package installer)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/ai-authenticity-detector.git
cd ai-authenticity-detector
```

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**

```bash
python -m venv venv
.env\Scriptsctivate
```

**On macOS & Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

# Training the Models

You must have the pre-trained model files in place before running the application. Follow these steps to generate them.

### A) Fake News Model (Local Training)

- **Get the Dataset:** Download the Fake and Real News Dataset from Kaggle.  
- **Place Files:** Extract and place `True.csv` and `Fake.csv` into the `fake_news_detector/dataset/` directory.  
- **Run Training Script:** Navigate to the `fake_news_detector` directory and run the training script.

```bash
cd fake_news_detector
python train_news_model.py
cd ..
```

This will generate `fn_model.pkl` and `fn_vectorizer.pkl` in the `fake_news_detector/saved_model/` folder.

### B) Deepfake Model (Kaggle Training Recommended)

Training this model is computationally expensive and is best done on a platform with free GPU access like Kaggle.

- **Get the Dataset:** Use the 140k Real and Fake Faces Dataset on Kaggle.  
- **Train on Kaggle:** Create a new Kaggle Notebook, attach the dataset, and enable the GPU Accelerator. Use the provided unified training script to train the model.  
- **Download the Model:** After the notebook finishes running, download the resulting `df_model.h5` file from the "Output" section of your notebook.  
- **Place the Model:** Place the downloaded `df_model.h5` file into the `deepfake_detector/saved_model/` directory on your local machine.

---

# Running the Application

Once both models are in their respective `saved_model` folders, you can launch the web application.

- Navigate to the `web_app` directory:

```bash
cd web_app
```

- Run the Flask application:

```bash
python app.py
```

- Open your web browser and go to the following URL:

```
http://127.0.0.1:5000
```

---

# Usage

- Navigate to the homepage.  
- Select the **Fake News Detector** tab to paste the text of a news article and click **Analyze Text**.  
- Select the **Deepfake Detector** tab to upload a video file (`.mp4`, `.mov`) and click **Analyze Video**. The page will show a loading state while processing.  
- The application will process the input and display the result on a new page.

---

# Limitations and Future Improvements

- **Dataset Bias:** The models are only as good as their training data. They may not perform as well on newer or different styles of misinformation.  
- **Performance:** Video analysis can be slow, as it processes frames sequentially. The current implementation samples frames to improve speed.  
- **No Real-time Face Detection:** The deepfake model processes the entire video frame. A future improvement would be to first detect and crop faces before analysis for higher accuracy.  
- **Model Sophistication:** More advanced models like BERT (for text) or Vision Transformers (for video) could yield better results.

---

# License

This project is licensed under the MIT License. See the LICENSE file for details.
