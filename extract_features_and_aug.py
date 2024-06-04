# NOISE
import os
import random
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)
    
# Function to load audio files and extract MFCC features and its super disgusting DONT DO IT LIKE THAT NEVER!!!
def extract_features_and_apply_aug(file_path):
    list_of_mfccs = []
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
    list_of_mfccs.append(mfccs)
    if random.uniform(0.0,1.0) > 0.5:
        noise_y = noise(y)
        mfccs = librosa.feature.mfcc(y=noise_y, sr=sr, n_mfcc=32)
        mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
        list_of_mfccs.append(mfccs)
    if random.uniform(0.0,1.0) > 0.5:
        stretch_y = stretch(y)
        mfccs = librosa.feature.mfcc(y=stretch_y, sr=sr, n_mfcc=32)
        mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
        list_of_mfccs.append(mfccs)
    if random.uniform(0.0,1.0) > 0.5:
        shift_y = shift(y)
        mfccs = librosa.feature.mfcc(y=shift_y, sr=sr, n_mfcc=32)
        mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
        list_of_mfccs.append(mfccs)
    if random.uniform(0.0,1.0) > 0.5:
        pitch_y = pitch(y, sr)
        mfccs = librosa.feature.mfcc(y=pitch_y, sr=sr, n_mfcc=32)
        mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
        list_of_mfccs.append(mfccs)
    
    return list_of_mfccs

# Path to the extracted dataset
print(os.listdir("."))
dataset_path = "./gtzan_genres/Data/genres_original/"
print(dataset_path)
genres = os.listdir(dataset_path)


# Lists to hold features and labels
train_features = []
test_features = []
test_labels = []
train_labels = []
# Load and preprocess the data
for genre in tqdm(genres):
    genre_path = os.path.join(dataset_path, genre)
    print(genre)
    for file in tqdm(os.listdir(genre_path)):
           try:
            file_path = os.path.join(genre_path, file)
            if random.uniform(0.0, 1.0)< 0.7:
                mfccs = extract_features_and_apply_aug(file_path)
                for i in mfccs:
                    train_features.append(i)
                for i in range(len(mfccs)):
                    train_labels.append(genre)
                # plt.figure(figsize=(10, 6))
                # librosa.display.specshow(mfccs, x_axis='time')
                # plt.colorbar()
                # plt.title('MFCC')
                # plt.tight_layout()
                # plt.show()
            else:
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
                mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
                test_features.append(mfccs)
                test_labels.append(genre)
                
           except Exception as e:
             print(f"Error loading {file_path}: {e}")


# Convert lists to numpy arrays

train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
label_classes = label_encoder.classes_
print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)
np.save("mfccs_features.npy",train_features)
np.save("mfccs_labels.npy",train_labels)
np.save("mfccs_test_features.npy",test_features)
np.save("mfccs_test_labels.npy",test_labels)
np.save("label_classes.npy", label_classes)