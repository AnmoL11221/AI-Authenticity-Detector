import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("All libraries imported successfully.")
print("TensorFlow Version:", tf.__version__)
print("GPU Available:", "Yes" if tf.config.list_physical_devices('GPU') else "No")
DATA_DIR = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test'
CATEGORIES = ['fake', 'real']
IMG_SIZE = 128
OUTPUT_PATH = '/kaggle/working/saved_model/'

training_data = []
print(f"\nStep 1: Reading images from '{DATA_DIR}' and resizing to {IMG_SIZE}x{IMG_SIZE}...")

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)
    for img_name in tqdm(os.listdir(path), desc=f"Processing '{category}' images"):
        try:
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

print("\nStep 2: Shuffling data and creating final arrays...")
np.random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
y = np.array(y)

print("Data preprocessing complete.")
print(f"Features (images) shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print("\nStep 3: Splitting data and building the CNN model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
print("\nStep 4: Starting model training...")
epochs = 10
batch_size = 64
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    batch_size=batch_size)

print("\nModel training finished.")
print("\nStep 5: Saving the final model...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save(os.path.join(OUTPUT_PATH, 'df_model.h5'))
print("\n=======================================================")
print(f"SUCCESS! Model saved to: {os.path.join(OUTPUT_PATH, 'df_model.h5')}")
print("You can now download the file from the 'Output' section of the notebook viewer.")
print("=======================================================")