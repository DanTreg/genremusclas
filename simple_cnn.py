from keras import models, layers, regularizers
import visualkeras
class CNN(models.Sequential):
    def __init__(self, num_classes, input_dims):
        super().__init__()
        self.add(layers.Input(shape=(13,1290,1)))
        self.add(layers.Conv2D(32, (3, 3), activation='relu', 
                               kernel_regularizer=regularizers.l2(0.001), padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        
        self.add(layers.Conv2D(64, (3, 3), activation='relu', 
                               kernel_regularizer=regularizers.l2(0.001), padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        
        self.add(layers.Conv2D(128, (3, 3), activation='relu', 
                               kernel_regularizer=regularizers.l2(0.001), padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        

        self.add(layers.Flatten())
        self.add(layers.Dropout(0.5))
        
        self.add(layers.Dense(128, activation='relu', 
                              kernel_regularizer=regularizers.l2(0.001)))
        self.add(layers.Dense(num_classes, activation='softmax'))
