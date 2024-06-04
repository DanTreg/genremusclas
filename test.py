import os
import random
import librosa
import numpy as np
from tqdm import tqdm
import resnet_18_keras_custom
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import time
import simple_cnn
# Lists to hold features and labels
print(os.listdir("."))
dataset_path = "./gtzan_genres/Data/genres_original/"
print(dataset_path)
genres = os.listdir(dataset_path)

features = []
labels = []
# Load and preprocess the data
for genre in tqdm(genres):
    genre_path = os.path.join(dataset_path, genre)
    print(genre)
    for file in tqdm(os.listdir(genre_path)):
           try:
            file_path = os.path.join(genre_path, file)
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
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


features = np.array(features)
labels = np.array(labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X = np.expand_dims(features, axis=-1)

model = simple_cnn.CNN(10, (13,1290,1))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.load_weights('genre_classification_model.h5')


test_loss, test_acc = model.evaluate(X, y, verbose=2)
print(f'Test accuracy: {test_acc}')