import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# --- STEP 1: IMPORT CUSTOM DATA ---
print("Importing your custom data...")
# Assuming X and y are the variable names in batch_preprocessing
from batch_preprocessing import X as X_custom, y as y_custom 

# --- STEP 2: LOAD MNIST & MERGE ---
print("Loading MNIST and merging datasets...")
(X_m_train, y_m_train), (X_m_test, y_m_test) = mnist.load_data()

# Combine MNIST Train and Test for a larger pool
X_mnist = np.concatenate([X_m_train, X_m_test], axis=0)
y_mnist = np.concatenate([y_m_train, y_m_test], axis=0)

# Reshape and Normalize BOTH (Ensures they are 0.0 to 1.0)
X_custom = np.array(X_custom).reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_mnist = X_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# If your custom labels are already one-hot, convert them back to integers for merging
if len(np.array(y_custom).shape) > 1:
    y_custom = np.argmax(y_custom, axis=1)

# --- THE BIG MERGE ---
X_final = np.concatenate([X_custom, X_mnist], axis=0)
y_final = np.concatenate([y_custom, y_mnist], axis=0)

# Synchronized Shuffle (Ensures images and labels don't get mixed up)
indices = np.random.permutation(len(X_final))
X_final = X_final[indices]
y_final = y_final[indices]

# Final Conversion to One-Hot
y_final = to_categorical(y_final, 10)

# --- STEP 3: THE GOLDEN SPLIT ---
# Now X_train will have 85% of ~91,555 images = 77,821 images
X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.15, random_state=42)

# --- STEP 4: DATA AUGMENTATION ---
datagen = ImageDataGenerator(
    rotation_range=12,
    zoom_range=0.15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    brightness_range=[0.8, 1.2]
)

# --- STEP 5: THE DEEP-SCAN ARCHITECTURE ---
def build_deep_scan_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Block 1: Edge Detection
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2: Complex Pattern Recognition
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3: Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_deep_scan_model()

# --- STEP 6: TRAINING WITH CHECKPOINTS ---
save_path = r'D:\3.ANSH\IT\AI UNDERSTANDING\NUMBER RECOGNITION\deep_scan_model.h5'

# Automatically saves the best version so you don't lose progress if it crashes
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print(f"\nTraining on {len(X_train)} images...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16), # Steps should now be ~2432
    epochs=13, 
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

print(f"✅ SUCCESS! Best model saved to: {save_path}")