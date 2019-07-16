import sys
import getopt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as save_tikz
import data
import model
import params
import cv2
import tensorflow as tf
import os

def moving_average(series, window_size, use_fraction=False):
	"""Smoothes a data series by applying a moving average.

	Args:
		series: list of data points
		window_size: amount of data points to use for mean calculation
		use_fraction: specifies if window_size is a fraction of the dataset

	Returns:
		smoothed: list of smoothed data points
	"""
	smoothed = []
	if use_fraction:
		window_size = int(np.floor(window_size * len(series)))
	for i in range(len(series)):
		range_ = max(i-window_size+1, 0)
		v = series[range_:i+1]
		smoothed.append(np.mean(v))
	return smoothed

if __name__ == "__main__":
	#plotting parameters
	moving_average_window_size = .1

	#first get command line arguments
	args = sys.argv[1:]
	unixOptions = "t:cp:"
	gnuOptions = ["train=", "ckpt", "predict="]
	#default params
	arg_train = "multi"
	arg_ckpt = None
	arg_predict = ""
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-t", "--train"):
			arg_train = val
		elif arg in ("-c", "--ckpt"):
			arg_ckpt = os.path.join(params.CKPT_PATH, params.CKPT_FILE)
		elif arg in ("-p", "--predict"):
			arg_predict = val

	#initialize objects for further use
	data = data.Data()
	model = model.Model()

	#get modelnet dataset splitted in training and testing set
	if params.DATASET_LOAD_DYNAMIC:
		dataset = data.get_dynamic_dataset(params.DATASET_PATH, one_hot=True, create_labels=True)
	else:
		dataset = data.get_dataset(params.DATASET_PATH, one_hot=True, create_labels=True)

	#get weights
	weights, biases = model.get_weights()


	if arg_train == "single":
		#create placeholders for input and output data
		x = tf.placeholder(tf.float32, (None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x")
		y = tf.placeholder(tf.float32, (None, params.N_CLASSES), name="y")

		#rough training of model: only one image
		model.train(x, y, deepcopy(dataset), weights, biases, arg_ckpt)

	elif arg_train == "multi":
		#group module training with multiview input
		x_mv = tf.placeholder(tf.float32, (None, params.N_VIEWS, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x_mv")
		y = tf.placeholder(tf.float32, (None, params.N_CLASSES), name="y")

		#multi_view_dataset = copy.deepcopy(set)
		#multi_view_dataset = data.single_to_multi_view(*dataset, params.N_VIEWS)
		if params.DATASET_LOAD_DYNAMIC:
			multi_view_dataset = dataset
		else:
			multi_view_dataset = data.single_to_multi_view(*dataset, params.N_VIEWS)
		#train_batch_loss, train_batch_accuracy, epochs_train_loss, epochs_train_accuracy, epochs_test_accuracy, epochs_test_loss, learning_rate
		train_batches_loss, train_batches_accuracy, epochs_train_loss, epochs_train_accuracy, epochs_test_accuracy, epochs_test_loss, learning_rate = model.train(x_mv, y, multi_view_dataset, weights, biases, arg_ckpt)

	if params.USE_PYPLOT:
		path = os.path.join(params.RESULTS_PATH, "models", os.path.basename(params.CKPT_PATH))
		if not os.path.isdir(path):
			os.makedirs(path)
		np.savez(os.path.join(path, "raw_data.npz"), train_batches_loss=train_batches_loss, train_batches_accuracy=train_batches_accuracy, epochs_train_accuracy=epochs_train_accuracy, epochs_train_loss=epochs_train_loss, epochs_test_accuracy=epochs_test_accuracy, epochs_test_loss=epochs_test_loss)

		plt.figure(0)
		plt.xlabel("Iteration")
		plt.ylabel("Loss")
		plt.plot(train_batches_loss, "g-", label="Loss", alpha=0.2)
		plt.plot(moving_average(train_batches_loss, moving_average_window_size, use_fraction=True), "g-", label="Loss MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "train_batches_loss.png"))
		save_tikz(os.path.join(path, "train_batches_loss.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(1)
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.plot(train_batches_accuracy, "g-", label="Accuracy", alpha=0.2)
		plt.plot(moving_average(train_batches_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "train_batches_accuracy.png"))
		save_tikz(os.path.join(path, "train_batches_accuracy.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(2)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(epochs_train_loss, "g-", label="Training Loss", alpha=1)
		#plt.plot(moving_average(epochs_train_loss, moving_average_window_size, use_fraction=True), "g-", label="Training Loss MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "epochs_train_loss.png"))
		save_tikz(os.path.join(path, "epochs_train_loss.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(3)
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(epochs_train_accuracy, "g-", label="Training Accuracy", alpha=1)
		#plt.plot(moving_average(epochs_train_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Training Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "epochs_train_accuracy.png"))
		save_tikz(os.path.join(path, "epochs_train_accuracy.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(4)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(epochs_test_loss, "g-", label="Testing Loss", alpha=1)
		#plt.plot(moving_average(epochs_test_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Testing Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "epochs_test_loss.png"))
		save_tikz(os.path.join(path, "epochs_test_loss.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(5)
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(epochs_test_accuracy, "g-", label="Testing Accuracy", alpha=1)
		#plt.plot(moving_average(epochs_test_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Testing Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "epochs_test_accuracy.png"))
		save_tikz(os.path.join(path, "epochs_test_accuracy.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(6)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(epochs_train_loss, "g-", label="Training Loss")
		plt.plot(epochs_test_loss, label="Testing Loss")
		#plt.plot(moving_average(epochs_test_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Testing Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "loss.png"))
		save_tikz(os.path.join(path, "loss.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		plt.figure(7)
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(epochs_train_accuracy, "g-", label="Training Accuracy")
		plt.plot(epochs_test_accuracy, label="Testing Accuracy")
		#plt.plot(moving_average(epochs_test_accuracy, moving_average_window_size, use_fraction=True), "g-", label="Testing Accuracy MA")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(path, "accuracy.png"))
		save_tikz(os.path.join(path, "accuracy.tikz"), figureheight="\\figureheight", figurewidth="\\figurewidth")

		#plt.show()