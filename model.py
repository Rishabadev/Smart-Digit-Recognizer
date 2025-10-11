from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import time 
from sklearn.ensemble import RandomForestClassifier



DATASET_PATH = '/content/drive/MyDrive/MLProject/Dataset/'
TARGET_SIZE = (64, 64)


name_to_digit = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9
}

''' --- 1. Data Loading, Preprocessing, and Labeling ---'''

X_raw = []
filenames = []
X_final_list = []
y_final_list = []

print("Starting data loading and preprocessing...")
start_time = time.time() 
for filename in os.listdir(DATASET_PATH):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(DATASET_PATH, filename)
        img = cv2.imread(file_path)

        if img is not None:
            fname_lower = filename.lower()
            label = None

            
            for word, digit in name_to_digit.items():
                if word in fname_lower:
                    label = digit
                    break

            
            if label is None:
                for d in range(10):
                    if f"{d}" in fname_lower:
                        label = d
                        break

            if label is not None:
              
                resized = cv2.resize(img, TARGET_SIZE)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                flat = rgb.flatten()
                X_final_list.append(flat)
                y_final_list.append(label)
            else:
                print(f"Label not found for: {filename}")
        else:
            print(f" Could not load image: {file_path}")
            



X_final = np.array(X_final_list, dtype=np.float32)
y_final = np.array(y_final_list)


del X_final_list
del y_final_list

print(f"Preprocessing complete in {time.time() - start_time:.2f}s.")
print(f"Final usable images: {len(X_final)}")
print(f"X_final shape: {X_final.shape}, dtype: {X_final.dtype}")
print(f"y_final shape: {y_final.shape}")

'''  2. Data Splitting and Scaling '''

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, stratify=y_final
)
from collections import Counter
print("Class Distribution in Training Set:", Counter(y_train))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)


del X_train
del X_test

print(f"\nTraining samples: {len(X_train_scaled)}")
print(f"Testing samples: {len(X_test_scaled)}")

augment = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

augmented_images = []
augmented_labels = []


from collections import Counter
class_counts = Counter(y_final)
max_count = max(class_counts.values())

print("Performing augmentation for underrepresented classes...")

for label in range(10):
    count = class_counts[label]
    if count < max_count:
        needed = max_count - count
        print(f"Augmenting class {label}: need {needed} more samples")

        class_images = [img for img, lbl in zip(X_final, y_final) if lbl == label]
        class_images = np.array(class_images).reshape((-1, 64, 64, 3))

        generated = 0
        for img in class_images:
            img = img.reshape((1, 64, 64, 3))  
            for batch in augment.flow(img, batch_size=1):
                augmented_images.append(batch[0].astype(np.uint8).flatten())
                augmented_labels.append(label)
                generated += 1
                if generated >= needed:
                    break
            if generated >= needed:
                break


X_final_aug = np.vstack([X_final] + [np.array(augmented_images)])
y_final_aug = np.hstack([y_final] + [np.array(augmented_labels)])

del X_final

print(" Data augmentation complete.")
print(f"New dataset size: {len(X_final_aug)}")
  
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


X_aug = X_final_aug.reshape(-1, 64, 64, 3).astype('float32') / 255.0
y_aug = to_categorical(y_final_aug, num_classes=10)


X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_final_aug
)

print(f"X_train_cnn: {X_train_cnn.shape}, y_train_cnn: {y_train_cnn.shape}")
print(f"X_test_cnn: {X_test_cnn.shape}, y_test_cnn: {y_test_cnn.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax') 
])

model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train_cnn, y_train_cnn,
    epochs=15,
    batch_size=64,
    validation_data=(X_test_cnn, y_test_cnn)
)
loss, acc = model.evaluate(X_test_cnn, y_test_cnn)
print(f" Test Accuracy: {acc:.4f}")

import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


y_pred_cnn = model.predict(X_test_cnn)
y_pred_labels = np.argmax(y_pred_cnn, axis=1)
y_true_labels = np.argmax(y_test_cnn, axis=1)


cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN Confusion Matrix')
plt.show()

print(classification_report(y_true_labels, y_pred_labels))

model.save('/content/drive/MyDrive/cnn_model.keras')
