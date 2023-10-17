from tensorflow import keras
from keras.datasets import mnist
from autoencoder import Autoencoder
import pathlib
import matplotlib.pyplot as plt


ROOT_DIR = pathlib.Path(__file__).parent.absolute()
CHECKPOINTS_DIR = ROOT_DIR / 'checkpoints'
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


autoencoder = Autoencoder()

autoencoder.compile(
	optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True, epsilon=1e-8),
	loss=keras.losses.MeanSquaredError(
		reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE
	),
)

try:
	autoencoder.fit(x_train, x_train,
									epochs=500,
									batch_size=2400,
									shuffle=True,
									validation_data=(x_test, x_test),
									callbacks=[
										# keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
										keras.callbacks.EarlyStopping(monitor='val_loss', patience=40),
									]
									)
except KeyboardInterrupt:
	pass


encoded_imgs = autoencoder.encoder(x_test[0:10]).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# display original and decoded images
REDUCED_SIZE = 14
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()


autoencoder.save('autoencoder')
