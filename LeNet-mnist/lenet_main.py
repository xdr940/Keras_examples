# import the modules we need
from keras import backend as K

from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from lenet import LeNet
import pickle


# parameters
NUM_EPOCH = 5
BATCH_SIZE = 64
VERBOSE = 1#verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2#20%的验证集
IMG_ROWS, IMG_COLS = 28, 28#20*20的图片大小
NB_CLASSES = 10#10种分类输出
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)#input_shape=(3,150, 150)是theano的写法，而tensorflow需要写出：(150,150,3)。

# load mnist dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
K.set_image_dim_ordering("th")  # channel first

# normalize the data
X_train = X_train.astype("float32") # uint8-->float32
X_test = X_test.astype("float32")
X_train /= 255 # 归一化到0~1区间
X_test /= 255
X_train = X_train.reshape(X_train.shape[0], 1, IMG_ROWS, IMG_COLS)
X_test = X_test.reshape(X_test.shape[0], 1, IMG_ROWS, IMG_COLS)


'''这里就用1%，省的太慢'''
X_train=X_train[0:500]
Y_train=Y_train[0:500]

X_test=X_test[0:100]
Y_test=Y_test[0:100]

print(X_train.shape[0], "train samples")
print(Y_test.shape[0], "test samples")

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# init the optimizer and model
model = LeNet.createLeNet(input_shape=INPUT_SHAPE, nb_class=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])


modelfile = 'modelweight_1percent.model' #神经网络权重保存
file_path_history = 'historyfile.bin'#保存history


if os.path.exists(modelfile):#如果存在之前训练的权重矩阵，载入模型
    print('载入模型参数')
    model.load_weights(modelfile)
else:#否则训练
    print('开始训练')
    history = model.fit(X_train, Y_train,
						batch_size=BATCH_SIZE,
						epochs=NUM_EPOCH,
						verbose=VERBOSE,
						validation_split=VALIDATION_SPLIT)
    # show the data in history
    model.save(modelfile)
    # save history file
    historyfile = open(file_path_history, 'wb')
    pickle.dump(history, historyfile)
    historyfile.close()

print('evaluate started')
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('evaluation 结果')
print("Test score:", score[0])
print("Test accuracy:", score[1])

if os.path.exists(file_path_history):#如果存在之前训练的history
    historyfile = open(file_path_history, 'rb')
    historyfile.seek(0)
    history = pickle.load(historyfile)
    fig = plt.figure(1, figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

