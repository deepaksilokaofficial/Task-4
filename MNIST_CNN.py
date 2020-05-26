from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
def baseline_model():
	model = Sequential()
	model.add(Dense(units=512, input_dim=28*28, activation='relu'))
	model.add(Dense(units=10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = baseline_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=0)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:  %.2f%%" % (scores[1]*100))
file1 = open("result.txt","w+")
file1.write(str(scores[1]*100))
file1.close()
