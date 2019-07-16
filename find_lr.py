import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import params
import data
import model
import matplotlib2tikz.save as tikz_save
import os

plt.rc('text', usetex=True)
plt.rcParams["font.family"] = ["Latin Modern Roman"]

def get_save_path(fill="", ckpt_file=None):
	"""Creates a path where to save plots that is related to the used checkpoint
	and predicted object.

	Args:
		fill: string to put into path
		ckpt_file: string of checkpoint file

	Returns:
		path: path where to save plots, i.e Results/Predictions/mn-sl-4-5-20/
	"""
	filename = None
	path_stats = os.path.join(params.RESULTS_PATH, "models")
	#get name of used model/checkpoint
	if ckpt_file is None:
		model_name = os.path.basename(params.CKPT_PATH)
	else:
		model_name = os.path.basename(os.path.dirname(ckpt_file))

	#join fill string if given
	if fill:
		model_name = "_".join([model_name, fill])

	path = os.path.join(path_stats, model_name)

	#if directory does not exist create it
	if not os.path.isdir(path):
		os.makedirs(path)

	return path

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
dataset = data.get_dynamic_dataset(params.DATASET_PATH, one_hot=True)
#if arg_multi:
	#dataset = data.single_to_multi_view(*dataset)

#get weights
weights, biases = model.get_weights()

if arg_multi:
	x = tf.placeholder(tf.float32, (None, params.N_VIEWS, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x")
else:
	x = tf.placeholder(tf.float32, (None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x")
y = tf.placeholder(tf.float32, (None, params.N_CLASSES), name="y")

train_batch_loss, train_batch_accuracy, epochs_test_accuracy, learning_rates = model.train(x, y, dataset, weights, biases, find_lr=True)

np.savez(os.path.join(get_save_path(), "optimal_learning_rate.npz"), train_batch_loss=train_batch_loss, learning_rates=learning_rates)

plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.plot(learning_rates, "g-")
plt.tight_layout()
plt.savefig(os.path.join(get_save_path(), "optimal_learning_rate_lr.png"))
tikz_save(os.path.join(get_save_path(), "optimal_learning_rate_lr.tikz"))

plt.figure()
plt.xlabel("Learning Rate")
plt.ylabel("$\partial$Loss")
plt.plot(learning_rates, np.gradient(train_batch_loss), "g-")
plt.xscale("log")
plt.tight_layout()
plt.savefig(os.path.join(get_save_path(), "optimal_learning_rate_dloss.png"))
tikz_save(os.path.join(get_save_path(), "optimal_learning_rate_dloss.tikz"))


plt.figure()
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.plot(learning_rates, train_batch_loss, "g-")
plt.xscale("log")
plt.tight_layout()
plt.savefig(os.path.join(get_save_path(), "optimal_learning_rate_loss.png"))
tikz_save(os.path.join(get_save_path(), "optimal_learning_rate_loss.tikz"))

plt.show()


