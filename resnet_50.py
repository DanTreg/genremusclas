from keras import models, layers, regularizers
from keras.applications import ResNet50
class ResNet50Custom(models.Sequential):
    def __init__(self, num_classes, input_dims):
        super().__init__()
        imported_model= ResNet50(include_top=False,
        input_shape=(input_dims),
        pooling='avg',classes=10,
        weights=None)
        self.add(layers.Input(shape=(input_dims)))
    
        self.add(imported_model)
        
        self.add(layers.Dense(512, activation='relu'))
        self.add(layers.Dense(num_classes, activation='softmax'))
