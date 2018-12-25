import keras
from keras.datasets import cifar10
from cnn import CNN
import os
import pickle
import matplotlib.pyplot as plt
#训练超参数设置
BATCH_SIZE=64
IMG_ROWS, IMG_COLS = 32, 32#的图片大小
NB_CLASSES = 10#10种分类输出
IMG_CHANNELS=3
INPUT_SHAPE = ( IMG_ROWS, IMG_COLS,IMG_CHANNELS)
VALIDATION_SPLIT = 0.2#20%的验证集
EPOCHS=10

# The data, shuffled and split between train and test sets:
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Convert class vectors to binary class matrices.
Y_train = keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NB_CLASSES)


'''人为削减'''
X_train=X_train[0:1000]
Y_train=Y_train[0:1000]
X_test=X_test[0:200]
Y_test=Y_test[0:200]


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)


# train the model using RMSprop
model = CNN.createCNN(INPUT_SHAPE,NB_CLASSES)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


modelfile = 'modelweight_10percent.model' #神经网络权重保存
file_path_history = 'historyfile.bin'#保存history，留着作图

if os.path.exists(modelfile):#如果存在之前训练的权重矩阵，载入模型
	print('载入模型参数')
	model.load_weights(modelfile)
else:
	print('训练')
	history = model.fit(X_train, Y_train,
						batch_size=BATCH_SIZE,
						epochs=EPOCHS,
						verbose=1,
						validation_split=VALIDATION_SPLIT)
	print('\n')
	model.save(modelfile)
	historyfile = open(file_path_history, 'wb')
	pickle.dump(history, historyfile)
	historyfile.close()



#测试结果
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




'''作图'''
if os.path.exists(file_path_history):#如果存在之前训练的history
    #载入文件
    historyfile=open(file_path_history,'rb')
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
