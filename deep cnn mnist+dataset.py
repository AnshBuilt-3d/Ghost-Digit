import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- STEP 1: IMPORT CUSTOM DATA ---
from batch_preprocessing import X as X_custom, y as y_custom 

# --- STEP 2: LOAD & MERGE (With Synchronized Shuffle) ---
print("Merging MNIST and Custom Data...")
(X_m_train, y_m_train), (X_m_test, y_m_test) = mnist.load_data()
X_mnist = np.concatenate([X_m_train, X_m_test], axis=0)
y_mnist = np.concatenate([y_m_train, y_m_test], axis=0)

# Reshape and Normalize
X_custom = np.array(X_custom).reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_mnist = X_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Fix y_custom if it's one-hot encoded already
if len(y_custom.shape) > 1:
    y_custom = np.argmax(y_custom, axis=1)

# Combine
X_final = np.concatenate([X_custom, X_mnist], axis=0)
y_final = np.concatenate([y_custom, y_mnist], axis=0)

# CRITICAL: Shuffle X and y together so labels stay with the right images
indices = np.random.permutation(len(X_final))
X_final = X_final[indices]
y_final = y_final[indices]

y_final = to_categorical(y_final, 10)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.15)

# --- STEP 3: THE DEEP-SCAN ARCHITECTURE ---
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- STEP 4: TRAIN ---
model = build_model()
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

print(f"\nTraining on {len(X_train)} images...")
model.fit(datagen.flow(X_train, y_train, batch_size=16), 
          epochs=15, validation_data=(X_val, y_val))

model.save(r'D:\3.ANSH\IT\AI UNDERSTANDING\NUMBER RECOGNITION\deep_scan_modelwithmnist1.h5')
