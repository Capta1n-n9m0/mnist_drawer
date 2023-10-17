from keras import layers
import keras
import pathlib


class Autoencoder(keras.Model):
	def __init__(self, latent_dims=100):
		super(Autoencoder, self).__init__()
		self.encoder = keras.Sequential([
			layers.InputLayer(input_shape=(28, 28, 1)),
			layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
			layers.Flatten(),
			layers.Dense(latent_dims, activation='sigmoid'),
		])
		self.decoder = keras.Sequential([
			layers.InputLayer(input_shape=(latent_dims,)),
			layers.Dense(7*7*4, activation='sigmoid'),
			layers.Reshape((7, 7, 4)),
			layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
			layers.UpSampling2D((2, 2)),
			layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
			layers.UpSampling2D((2, 2)),
			layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
			layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
		])
	
	def call(self, x, training=False, mask=None, *args, **kwargs):
		encoded = self.encoder(x, training=training, mask=mask, *args, **kwargs)
		decoded = self.decoder(encoded, training=training, mask=mask, *args, **kwargs)
		return decoded
	
	def summary(self, *args, **kwargs):
		self.encoder.summary(*args, **kwargs)
		self.decoder.summary(*args, **kwargs)
		super(Autoencoder, self).summary(*args, **kwargs)
	
	def save(self, path, *args, **kwargs):
		path = pathlib.Path(path)
		path.mkdir(parents=True, exist_ok=True)
		self.encoder.save(path / 'encoder', *args, **kwargs)
		self.decoder.save(path / 'decoder', *args, **kwargs)
	
	@classmethod
	def load(cls, path):
		path = pathlib.Path(path)
		encoder = keras.models.load_model(path / 'encoder')
		decoder = keras.models.load_model(path / 'decoder')
		model = cls()
		model.encoder = encoder
		model.decoder = decoder
		return model
		
if __name__ == "__main__":
	autoencoder = Autoencoder()
	autoencoder.build((None, 28, 28))
	autoencoder.summary()
	