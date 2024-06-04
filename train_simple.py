import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import TensorBoard, EarlyStopping
import simple_cnn
features = np.load("mfccs_features.npy")
labels = np.load("mfccs_labels.npy")
features_test = np.load("mfccs_test_features.npy")
labels_test = np.load("mfccs_test_labels.npy")
# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y_test = label_encoder.transform(labels_test)
# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.3, random_state=42)

# Ensure the data shape is appropriate for the model
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(features_test, axis=-1)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(16,drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(16, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define log directory for TensorBoard
log_dir = "logs/fit/"


# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Define the CNN model
model = simple_cnn.CNN(10, (13,1290,1))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()
# Train the model
earlystopping = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=20,
                                        restore_best_weights=True)
history = model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[tensorboard_callback, earlystopping])



# Save the trained model
model.save('genre_classification_model.h5')

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')