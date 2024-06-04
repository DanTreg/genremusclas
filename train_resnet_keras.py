import numpy as np
import librosa
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import AdamW
from resnet_50 import ResNet50Custom
import soundfile as sf

features = np.load("mfccs_features.npy")
labels = np.load("mfccs_labels.npy")
features_test = np.load("mfccs_test_features.npy")
labels_test = np.load("mfccs_test_labels.npy")
# Encode labels

# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)

# Ensure the data shape is appropriate for the model
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(features_test, axis=-1)


# Augment the training data
X_train_augmented, y_train_augmented = X_train, y_train



# Define log directory for TensorBoard
log_dir = "logs/fit/"

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Define the CNN model

model = ResNet50Custom(10, (32,1290,1))


#opt = AdamW(learning_rate=1e-4,weight_decay=5e-4) #parameters suggested by Nick!!!
model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics=["accuracy"]) 


train_dataset = tf.data.Dataset.from_tensor_slices((X_train_augmented, y_train_augmented))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, labels_test))

train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(16,drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(16, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)


model.summary()
earlystopping = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=20,
                                        restore_best_weights=True)
history = model.fit(train_dataset, epochs=200, validation_data=val_dataset,callbacks=[earlystopping, tensorboard_callback])
model.save('genre_classification_model.h5')
test_loss, test_acc = model.evaluate(X_test, labels_test, verbose=2)
print(f'Test accuracy: {test_acc}')