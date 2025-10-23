import os
import pathlib
import numpy as np
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')

import keras
from keras import layers, callbacks, models
print(f'Using Keras {keras.__version__}')
try:
    import mlflow
    import mlflow.keras
    mlflow.set_tracking_uri(MLFLOW_URI)
except Exception as e:
    mlflow = None



def load_data(create_sample=False):
    req = ['data/X_train.npy','data/y_train.npy','data/X_test.npy','data/y_test.npy']
    if create_sample and any(not pathlib.Path(p).exists() for p in req):
        print('Creating synthetic data...')
        pathlib.Path('data').mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        X_train = (np.random.rand(50, 128, 128, 3) * 255).astype(np.float32)
        y_train = np.array([f'class_{i % 5}' for i in range(50)])
        X_test = (np.random.rand(10, 128, 128, 3) * 255).astype(np.float32)
        y_test = np.array([f'class_{i % 5}' for i in range(10)])
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)
    
    X_train = np.load('data/X_train.npy', allow_pickle=True).astype(np.float32)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True).astype(np.float32)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    
    assert X_train.ndim == 4 and X_train.shape[1:] == (128, 128, 3)
    assert X_test.ndim == 4 and X_test.shape[1:] == (128, 128, 3)
    print('Data shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def build_model(num_classes, light=False):
    if light:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    base = keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet', pooling='avg')
    base.trainable = False
    inputs = keras.Input(shape=(128, 128, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-sample', action='store_true')
    parser.add_argument('--light-model', action='store_true')
    parser.add_argument('--no-mlflow', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    print(f'Keras: {keras.__version__}')
    
    X_train, y_train, X_test, y_test = load_data(create_sample=args.create_sample)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    # Ensure y_test uses the same classes as y_train
    try:
        y_test_enc = le.transform(y_test)
    except ValueError as e:
        # If test set has unseen labels, only use classes present in both sets
        print(f'⚠️  Test set contains unseen labels: {e}')
        common_indices = [i for i, label in enumerate(y_test) if label in le.classes_]
        if len(common_indices) == 0:
            print('⚠️  No overlap between train and test classes. Using train set for evaluation.')
            X_test = X_train
            y_test_enc = y_train_enc
        else:
            print(f'Using {len(common_indices)} test samples with seen classes.')
            X_test = X_test[common_indices]
            y_test = y_test[common_indices]
            y_test_enc = le.transform(y_test)
    
    num_classes = len(le.classes_)
    
    model = build_model(num_classes, light=args.light_model)
    
    history = model.fit(X_train, y_train_enc, validation_data=(X_test, y_test_enc), epochs=args.epochs, batch_size=32, verbose=1)
    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f'Test accuracy: {acc:.4f}')
    
    # Always save a .keras file for Keras 3.x compatibility
    keras_path = 'models/model.keras'
    model.save(keras_path)
    print(f'Saved model to {keras_path}')
    
    print('Training complete!')

if __name__ == '__main__':
    main()
