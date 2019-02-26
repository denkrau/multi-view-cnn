import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import globals
import data
import model

#get and evaluate command line parameters
args = sys.argv[1:]
arg_multi = True
for arg in args:
	if arg in ("-s"):
		arg_multi = False

#create objects for further use
model = model.Model()
data = data.Data()

#get modelnet dataset splitted in training and testing set
dataset = data.get_dataset(globals.DATASET_PATH, one_hot=True)
if arg_multi:
	dataset = data.single_to_multi_view(*dataset)

#get weights
weights, biases = model.get_weights()

if arg_multi:
	x = tf.placeholder(tf.float32, (None, globals.N_VIEWS, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1), name="x")
else:
	x = tf.placeholder(tf.float32, (None, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1), name="x")
y = tf.placeholder(tf.float32, (None, globals.N_CLASSES), name="y")

iter_loss, learning_rates, _, _ = model.train(x, y, dataset, weights, biases, find_lr=True)

plt.subplot(1,2,1)
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.plot(learning_rates)
plt.subplot(1,2,2)
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.plot(learning_rates, iter_loss)
plt.xscale("log")
plt.tight_layout()
plt.show()


