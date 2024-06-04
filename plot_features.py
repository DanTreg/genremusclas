import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import random
features = np.load("mfccs_features.npy")
labels = np.load("mfccs_labels.npy")
test_features = np.load("mfccs_test_features.npy")
test_labels = np.load("mfccs_test_labels.npy")

plt.figure(figsize = (10, 4))
sns.countplot(y = labels, palette = 'viridis')
plt.title('Distribution of Classes for training', fontsize = 16)
plt.xlabel('Count', fontsize = 14)
plt.ylabel('Class', fontsize = 14)

plt.figure(figsize = (10, 4))
sns.countplot(y = test_labels, palette = 'viridis')
plt.title('Distribution of Classes for testing', fontsize = 16)
plt.xlabel('Count', fontsize = 14)
plt.ylabel('Class', fontsize = 14)

for i in range(2):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(random.choice(features), x_axis='time')
    plt.colorbar()
    plt.title('MFCC of training including augmentations')
    plt.tight_layout()
for i in range(2):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(random.choice(test_features), x_axis='time')
    plt.colorbar()
    plt.title('MFCC of test without augmentations')
    plt.tight_layout()

plt.show()

