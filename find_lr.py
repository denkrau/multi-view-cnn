import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import globals
import data
import model

model = model.Model()
data = data.Data()

#get modelnet dataset splitted in training and testing set
dataset = data.get_dataset(globals.DATASET_PATH, one_hot=True)

#get weights
weights, biases = model.get_weights()

x = tf.placeholder(tf.float32, (None, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1), name="x")
y = tf.placeholder(tf.float32, (None, globals.N_CLASSES), name="y")

iter_loss, learning_rates, _, _ = model.train(x, y, dataset, weights, biases, find_lr=True)

plt.subplot(1,2,1)
plt.plot(learning_rates)
plt.subplot(1,2,2)
plt.plot(learning_rates, iter_loss)
plt.xscale("log")
plt.show()


