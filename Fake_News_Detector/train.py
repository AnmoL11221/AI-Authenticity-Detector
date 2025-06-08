import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATASET_PATH = 'dataset/'
TRUE_NEWS_FILE = os.path.join(DATASET_PATH, 'True.csv')
FAKE_NEWS_FILE = os.path.join(DATASET_PATH, 'Fake.csv')
SAVE_PATH = 'saved_models/'
MODEL_NAME = 'fn_model.pkl'
VECTORIZER_NAME = 'fn_vectorizer.pkl'

def train_and_save_model():
    print("Step 1: Loading and preprocessing data...")

    if not os.path.exists(TRUE_NEWS_FILE) or not os.path.exists(FAKE_NEWS_FILE):
        print(f"Error: Dataset files not found in '{DATASET_PATH}' directory.")
        print("Please download the dataset from Kaggle and place True.csv and Fake.csv in the dataset folder.")
        return

    df_true = pd.read_csv(TRUE_NEWS_FILE)
    df_fake = pd.read_csv(FAKE_NEWS_FILE)

    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'

    df = pd.concat([df_true, df_fake])

    df['full_text'] = df['title'] + ' ' + df['text']
    df = df.drop(columns=['title', 'text', 'subject', 'date'])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Data loaded successfully.")
    print(f"Total articles: {len(df)}")
    print("Label distribution:\n", df['label'].value_counts())
    print("-" * 20)

    print("Step 2: Performing feature extraction with TF-IDF...")

    X = df['full_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    tfidf_test = tfidf_vectorizer.transform(X_test)
    print("Text data vectorized.")
    print("-" * 20)

    print("Step 3: Training the Passive-Aggressive Classifier...")
    pac = PassiveAggressiveClassifier(max_iter=100, C=0.5, random_state=42)
    pac.fit(tfidf_train, y_train)

    print("Model training complete.")
    print("-" * 20)

    print("Step 4: Evaluating the model...")
    y_pred = pac.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print("\nConfusion Matrix:")
    print(cm)
    print("-" * 20)

    print("Step 5: Saving the model and vectorizer to disk...")

    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, MODEL_NAME), 'wb') as model_file:
        pickle.dump(pac, model_file)
    with open(os.path.join(SAVE_PATH, VECTORIZER_NAME), 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

    print(f"Model saved to: {os.path.join(SAVE_PATH, MODEL_NAME)}")
    print(f"Vectorizer saved to: {os.path.join(SAVE_PATH, VECTORIZER_NAME)}")
    print("\nTraining process finished successfully!")


if __name__ == '__main__':
    train_and_save_model()