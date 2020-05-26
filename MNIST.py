from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = mnist.load_data()
unit1 = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], unit1)
X_test = X_test.reshape(X_test.shape[0], unit1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
unit2=y_test.shape[1]
def baseline_model():
	model = Sequential()
	model.add(Dense(unit1, input_dim=unit1, activation='relu'))
	model.add(Dense(unit, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=0)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:  %.2f%%" % (scores[1]*100))
file1 = open("result.txt","w+")
file1.write(str(scores[1]*100))
file1.close()
