import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# === Configuration ===
RAW_DIR = Path("data/raw_images")
PROCESSED_DIR = Path("data/processed_images")
IMAGE_SIZE = (128, 128)  # resize all images to 128x128 pixels
ALLOWED_EXT = (".jpg", ".jpeg", ".png")

# === Helper: Create folder if not exists ===
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_images():
    """Load image paths and labels from the raw_images directory."""
    images, labels = [], []
    for class_folder in RAW_DIR.iterdir():
        if class_folder.is_dir():
            label = class_folder.name
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in ALLOWED_EXT:
                    images.append(str(img_path))
                    labels.append(label)
    return images, labels

def preprocess_image(img_path):
    """Load and process a single image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # normalize
    return img

def preprocess_all_images():
    """Process all images and save numpy arrays for training."""
    images, labels = load_images()
    X, y = [], []

    for i, img_path in enumerate(images):
        img = preprocess_image(img_path)
        if img is not None:
            X.append(img)
            y.append(labels[i])
            # Save processed image visually
            save_path = PROCESSED_DIR / Path(img_path).name
            cv2.imwrite(str(save_path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    X = np.array(X)
    y = np.array(y)
    print(f"✅ Processed {len(X)} images. Shape: {X.shape}")
    return X, y

def split_data(X, y):
    """Split dataset into train/test sets."""
    # Check if stratification is possible (each class needs at least 2 samples)
    from collections import Counter
    class_counts = Counter(y)
    min_count = min(class_counts.values()) if class_counts else 0
    
    # If any class has fewer than 2 samples, skip stratification
    if min_count < 2:
        print(f"⚠️  Warning: Some classes have <2 samples (min={min_count}). Skipping stratification.")
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    X, y = preprocess_all_images()
    X_train, X_test, y_train, y_test = split_data(X, y)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    print("📁 Saved processed data arrays to /data/")
