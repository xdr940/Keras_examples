'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
from cnnnet import CNN
import os
import matplotlib.pyplot as plt
import pickle #存储对象

'''输入参数'''
batch_size = 64
num_classes = 10
epochs = 10
# input image dimensions
img_rows, img_cols = 28, 28


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':#由于历史原因，channel在最前和最后可能不确定
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#数据预处理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255#使得在0~1
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

'''这里为了调试方便，只要1/10'''
x_train = x_train[0:500]
y_train=y_train[0:500]
x_test=x_test[0:100]
y_test=y_test[0:100]



'''网络模型建立'''
model=CNN.creatCNN(input_shape,num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

modelfile = 'modelweight_10percent.model' #神经网络权重保存
file_path_history = 'historyfile.bin'#保存history

if os.path.exists(modelfile):#如果存在之前训练的权重矩阵，载入模型
    print('载入模型参数')
    model.load_weights(modelfile)
else:#否则训练
    print('开始训练')
    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('\n')
    # show the data in history
    model.save(modelfile)
    #save history file
    historyfile = open(file_path_history, 'wb')
    pickle.dump(history, historyfile)
    historyfile.close()



#测试结果
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




'''作图'''
if os.path.exists(file_path_history):#如果存在之前训练的history
    historyfile=open(file_path_history,'rb')
    #historyfile.read()
    historyfile.seek(0)
    history = pickle.load(historyfile)

    fig = plt.figure(1, figsize=(10, 5))

    plt.subplot(1,2,1)
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
