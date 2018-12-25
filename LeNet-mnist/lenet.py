from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
# define the Sequential model
class LeNet:

    @staticmethod
    def createLeNet(input_shape, nb_class):
        feature_layers = [
            Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(50, kernel_size=5, padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten()
        ]

        classification_layers = [
            Dense(500),
            Activation("relu"),
            Dense(nb_class),
            Activation("softmax")
        ]
		
		
        model = Sequential(feature_layers + classification_layers)
        model.summary()#打印出模型概述信息
        return model
    def creteLeNet2(input_shape, nb_class):#不同的写法
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(nb_class))
        model.add(Activation("softmax"))

        model.summary()
        return model


