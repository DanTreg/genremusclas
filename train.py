import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import TensorBoard

# Function to load audio files and extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

# Path to the extracted dataset
print(os.listdir("."))
dataset_path = "./gtzan_genres/Data/genres_original/"
print(dataset_path)
genres = os.listdir(dataset_path)


# Lists to hold features and labels
features = []
labels = []
# Load and preprocess the data
for genre in tqdm(genres):
    genre_path = os.path.join(dataset_path, genre)
    print(genre)
    for file in tqdm(os.listdir(genre_path)):
          try:
            file_path = os.path.join(genre_path, file)
            mfccs = extract_features(file_path)
            mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
            features.append(mfccs)
            labels.append(genre)
            # plt.figure(figsize=(10, 6))
            # librosa.display.specshow(mfccs, x_axis='time')
            # plt.colorbar()
            # plt.title('MFCC')
            # plt.tight_layout()
            # plt.show()
          except Exception as e:
            print(f"Error loading {file_path}: {e}")


# Convert lists to numpy arrays

features = np.array(features)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Ensure the data shape is appropriate for the model
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)



def augment_data(X, y):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        # Apply time stretching
        time_stretch = librosa.effects.time_stretch(X[i].squeeze(), rate=0.8)
        if time_stretch.shape[1] == 1290:
            augmented_X.append(time_stretch[..., np.newaxis])
            augmented_y.append(y[i])
        # Apply pitch shifting
        pitch_shift = librosa.effects.pitch_shift(X[i].squeeze(), sr=22050, n_steps=2)
        if pitch_shift.shape[1] == 1290:
            augmented_X.append(pitch_shift[..., np.newaxis])
            augmented_y.append(y[i])
        # Add white noise
        noise = np.random.normal(0, 0.1, X[i].shape)
        noisy = X[i] + noise
        if noisy.shape[1] == 1290:
            augmented_X.append(noisy)
            augmented_y.append(y[i])
    return np.array(augmented_X), np.array(augmented_y)

# Augment the training data
X_train_augmented, y_train_augmented = augment_data(X_train, y_train)



# Define log directory for TensorBoard
log_dir = "logs/fit/"

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(13, 1290, 1), kernel_regularizer=regularizers.l2(0.001), padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(len(genres), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_augmented, y_train_augmented, epochs=50, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

# Save the trained model
model.save('genre_classification_model.h5')

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')