from autoencoder import Autoencoder

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

autoencoder = Autoencoder.load('autoencoder')

encoded_imgs = autoencoder.encoder(x_test[0:10]).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# display original, encoded and decoded images

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
	# display original
	ax = plt.subplot(3, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# display encoded
	ax = plt.subplot(3, n, i + 1 + n)
	plt.imshow(encoded_imgs[i].reshape(10, 10))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# display reconstruction
	ax = plt.subplot(3, n, i + 1 + n + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
plt.show()