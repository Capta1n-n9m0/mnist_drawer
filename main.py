import pygame
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
# Y to categorical
Y = keras.utils.to_categorical(Y, 10)
# Y = np.expand_dims(Y, axis=2)

model = Sequential(
  [
    layers.InputLayer(input_shape=(10,)),  # Input shape for the digit (0-9)
    layers.Dense(128, activation='relu'),
    layers.Reshape((1, 1, 128)),  # Reshape for convolution
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='valid'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
  ]
)

# Compile the model with mean squared error (MSE) loss
model.compile(
  optimizer=keras.optimizers.Adam(),
  loss=keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy', 'mse']
)

model.summary()

# exit(1)

# Print the model summary
# model.summary()

# model.fit(Y, X, epochs=10, batch_size=32)

DIGITS = np.array([
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 7
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
])

# ps = model.predict(DIGITS)
# for i in range(10):
#   plt.imsave('before' + str(i) + '.png', ps[i, :, :, 0], cmap='gray')
#
# # fit only 10 samples
# # model.fit(Y[0:10], X[0:10], epochs=2, batch_size=10)
#
# model.fit(Y, X, epochs=2)
#
# ps = model.predict(DIGITS)
# for i in range(10):
#   plt.imsave('after' + str(i) + '.png', ps[i, :, :, 0], cmap='gray')





class GUI:
  def __init__(self):
    self.screen = pygame.display.set_mode((1000, 1000))
    self.running = True
    self.screen.fill((0, 0, 0))
    self.step = 0
    self.batch_size = 70
    pygame.display.set_caption("Digits Learner")
    pygame.display.flip()
  
  def run(self):
    while self.running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.running = False
      self.draw_digits()
      self.learn()
    pygame.quit()
  
  def draw_digits(self):
    ps = model.predict(DIGITS, verbose=0)
    # draw prediction on screen
    for i in range(10):
      img = Image.fromarray((np.array(ps[i, :, :, 0]) * 255).astype(np.uint8), 'L')
      img = img.resize((200, 200))
      img = np.array(img)
      img = np.fliplr(img)
      img = np.rot90(img, 1)
      img = np.stack((img, img, img), axis=2)
      img = pygame.surfarray.make_surface(img)
      self.screen.blit(img, (i * 200 % 1000, i * 200 // 1000 * 200))
      
    pygame.display.flip()
  
  def learn(self):
    print(self.step)
    start = self.step * self.batch_size
    stop = start + self.batch_size
    if(stop >= len(X)):
      stop = len(X)
      self.step = 0
    else:
      self.step += 1
    model.fit(Y[start:stop], X[start:stop], epochs=1, batch_size=self.batch_size, verbose=0)


def main():
  gui = GUI()
  gui.run()


if __name__ == "__main__":
  main()
