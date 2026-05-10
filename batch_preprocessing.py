import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

#step 1
# Your base directory
BASE_DIR = r'D:\3.ANSH\IT\AI UNDERSTANDING\NUMBER RECOGNITION\archive (17)'

# These will hold our raw info
file_paths = []
labels = []

# We manually check each folder from 0 to 9
for digit in range(10):
    folder_path = os.path.join(BASE_DIR, str(digit))
    
    # Get every file in this specific folder
    files_in_folder = os.listdir(folder_path)
    
    for filename in files_in_folder:
        if filename.endswith(".jpg"):
            # Create the full path: D:\...\0\Zero_full (1).jpg
            full_path = os.path.join(folder_path, filename)
            
            file_paths.append(full_path)
            labels.append(digit)

print(f"Step 1 Complete: Found {len(file_paths)} image paths.")


#step 2

processed_images = []

print("Step 2: Starting image processing... this might take a minute.")

for path in file_paths:
    # 1. Load the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Invert (Black ink -> White light)
    img = cv2.bitwise_not(img)
    
    # 3. Resize to 28x28 (The standard size for AI)
    img = cv2.resize(img, (28, 28))
    
    # 4. Normalize (Convert 0-255 to 0.0-1.0)
    img = img.astype('float32') / 255.0
    
    # Add it to our master list
    processed_images.append(img)

print("Step 2 Complete: All images processed.")


#step 3

# Convert lists to NumPy arrays
X = np.array(processed_images)
y = np.array(labels)

# Reshape X to (Total Images, Height, Width, Channels)
# Since it's grayscale, channels = 1
X = X.reshape(-1, 28, 28, 1)

# Convert labels (0-9) into categories
# Example: 3 becomes [0,0,0,1,0,0,0,0,0,0]
y = to_categorical(y, num_classes=10)

print("Step 3 Complete: Data is ready for the CNN.")
print(f"Final Image Shape: {X.shape}")
print(f"Final Label Shape: {y.shape}")