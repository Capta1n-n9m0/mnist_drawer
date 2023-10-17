import keras
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np

drawer = keras.models.load_model('drawer')
drawer.summary()

# draw mixes between 2 and 8
# generate 10 points between 2 and 8
x = np.zeros((10, 10))
x[:, 2] = np.linspace(0, 1, 10)
x[:, 8] = np.linspace(1, 0, 10)
x = tf.convert_to_tensor(x, dtype=tf.float32)

# draw the points
fig, axs = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
	axs[i].imshow(drawer.predict(x[i:i+1])[0])
	axs[i].axis('off')

plt.show()


