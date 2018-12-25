from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense,Dropout


class CNN:
	def creatCNN(input_shape,num_classes):

		model = Sequential()
		model.add(Conv2D(filters=32, kernel_size=(3, 3),padding="same",input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(filters=63, padding="same",kernel_size=(3, 3)))#???咋回事64-》63就好了
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))

		#model.add(Dropout(0.5))
		model.add(Dense(num_classes))
		model.add(Activation("softmax"))

		model.summary()
		return model


