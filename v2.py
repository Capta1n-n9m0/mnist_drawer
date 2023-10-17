from autoencoder import Autoencoder
import keras
from keras.datasets import mnist
from keras import layers
import tensorflow as tf

autoencoder = Autoencoder.load('autoencoder')
autoencoder.build((None, 28, 28))

encoder = autoencoder.encoder
decoder = autoencoder.decoder

# freeze the decoder layers
for layer in decoder.layers:
	layer.trainable = False

# freeze the encoder layers
for layer in encoder.layers:
	layer.trainable = False

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
enc_x_train = encoder.predict(x_train, batch_size=6000)
x_test = x_test.astype('float32') / 255.0
enc_x_test = encoder.predict(x_test, batch_size=6000)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

coder = keras.Sequential([
	layers.Input(shape=(10,)),
	layers.Dense(128, activation='relu'),
	layers.Dense(256, activation='relu'),
	layers.Dense(256, activation='relu'),
	layers.Dense(128, activation='relu'),
	# as the encoder output is 100, we need to add a dense layer to match the decoder input
	layers.Dense(decoder.input_shape[1], activation='sigmoid'),
])

coder.compile(optimizer=keras.optimizers.Adam(
	learning_rate=0.001,
	amsgrad=True,
	epsilon=1e-8,
), loss='mse')
coder.build((None, 10))
coder.summary()

coder.fit(y_train, enc_x_train, epochs=500, batch_size=6000, validation_data=(y_test, enc_x_test))

drawer = keras.Sequential([
	layers.Input(shape=(10,)),
	coder,
	decoder,
])

drawer.compile(optimizer='adam', loss='mse')
drawer.build((None, 10))
drawer.summary()

import matplotlib.pyplot as plt

# draw each digit
for i in range(10):
	plt.subplot(2, 5, i + 1)
	plt.imshow(drawer(tf.one_hot([i], 10))[0])
	plt.axis('off')
plt.show()

drawer.save('drawer')
