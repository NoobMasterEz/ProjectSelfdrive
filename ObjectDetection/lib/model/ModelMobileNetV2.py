import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import os
import pathlib

class MobilenetV2(object):
    """
    Arguments: 
        input_shape : 	Optional shape tuple, to be specified if you would like to use a model with an input image resolution that is not (224, 224, 3). It should have exactly 3 inputs channels (224, 224, 3). You can also omit this option if you would like to infer input_shape from an input_tensor. If you choose to include both input_tensor and input_shape then input_shape will be used if they match, if the shapes do not match then we will throw an error. E.g. (160, 160, 3) would be one valid value.
        alpha :	Float between 0 and 1. controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper, but the name is kept for consistency with applications.MobileNetV1 model in Keras.
                If alpha < 1.0, proportionally decreases the number of filters in each layer.
                If alpha > 1.0, proportionally increases the number of filters in each layer.
                If alpha = 1, default number of filters from the paper are used at each layer.
        include_top : Boolean, whether to include the fully-connected layer at the top of the network. Defaults to True.
                weights	String, one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
        input_tensor : Optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        pooling	: String, optional pooling mode for feature extraction when include_top is False.
                None means that the output of the model will be the 4D tensor output of the last convolutional block.
                avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
                max means that global max pooling will be applied.
        classes : Integer, optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
        classifier_activation: A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True. Set classifier_activation=None to return the logits of the "top" layer.
    """
    def __init__(self,**kawge):
        
        
        self.input_shape=kawge["input_shape"]
        self.alpha=kawge["alpha"]
        self.include_top=kawge["include_top"]
        self.weights=kawge["weights"]
        self.classifier_activation=kawge["classifier_activation"]
        
    @property
    def Build(self):
        
        self.model= tf.keras.applications.MobileNetV2(self.input_shape,self.alpha,self.include_top,self.weights,self.weights)
        self.model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        self.model.summary()
        
    def Save_model(self,path):
        self.model.Save(path)

    def Train(self):
        pass 

class TranferLearning(object):

    def __init__(self):
        pass

    @property
    def __LoadModel(self):
        """
        load model for tranfer learning
        """
        self.Base_model = tf.keras.models.load_model("Save/")
        


    def fit(self):
        pass 

    def BuildModel(self,numclass):
        self.__LoadModel
        
        self.Base_model.layers.pop()
        for layer in self.Base_model.layers[:-4]:
            layer.trainable = False

        output= tf.keras.layers.Dense(numclass, activation="softmax")
        output = output(self.Base_model.layers[-1].output)
        model = tf.keras.Model(inputs=self.Base_model.inputs, outputs=output)
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

def load_data(IMAGE_SHAPE):
    """This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset"""
    # download the dataset and extract it
    data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)
    # count how many images are there
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Number of images:", image_count)
    # get all classes for this dataset (types of flowers) excluding LICENSE file
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    # roses = list(data_dir.glob('roses/*'))
    # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    # make the training dataset generator
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="training")
    # make the validation dataset generator
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size, 
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="validation")
    return train_data_gen, test_data_gen, CLASS_NAMES

if __name__ == "__main__":
    
 
    """
    model=MobilenetV2(input_shape=(224,224,3),include_top=False, weights=None,alpha=1.0,classifier_activation="softmax")
    model.Tranfer()
    """
    batch_size = 32
    # 5 types of flowers
    num_classes = 5
    # training for 10 epochs
    epochs = 10
    # size of each image
    IMAGE_SHAPE = (224, 224, 3)

    train_generator, validation_generator, class_names = load_data(IMAGE_SHAPE)
    model=MobilenetV2(input_shape=(224,224,3),include_top=False, weights=None,alpha=1.0,classifier_activation="softmax")
    model_name = "MobileNetV2_finetune_last5"
    tensorboard=tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint =tf.keras.callbacks.ModelCheckpoint(os.path.join("results", f"{model_name}" + "-loss-{val_loss:.2f}.h5"),
                                save_best_only=True,
                                verbose=1)
    if not os.path.isdir("results"):
        os.mkdir("results")
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
    model.fit(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint])