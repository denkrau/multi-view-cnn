import sys
import getopt
import os
import numpy as np
import cv2
import tensorflow as tf
import model
import data
import params
import matplotlib.pyplot as plt
import matplotlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.rcParams["savefig.directory"] = os.path.join(os.path.dirname(os.getcwd()), "Results")

if __name__ == "__main__":
	args = sys.argv[1:]
	unixOptions = "s"
	gnuOptions = ["single"]
	arg_multi = True
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-s", "--single"):
			arg_multi = False

	#initialize objects for further use
	model = model.Model()
	data = data.Data()

	x_train, y_train, x_test, y_test = data.get_dataset(params.DATASET_PATH, one_hot=True)
	n_objects_train, n_objects_test = data.get_amount_objects_per_category()

	if arg_multi:
		x_train, y_train, x_test, y_test = data.single_to_multi_view(x_train, y_train, x_test, y_test, params.N_VIEWS)
	labels = data.load_labels()

	_, _, _, _, _, correct_predictions = model.predict(x_test, y_test)

	if params.DATASET_IS_SINGLELABEL:
		overall_accuracy = []
		print("\n*** ACCURACY PER LABEL ***")
		for i in range(params.DATASET_NUMBER_CATEGORIES):
			for j in range(params.DATASET_NUMBER_MATERIALS):
				label_id = i * params.DATASET_NUMBER_MATERIALS + j
				label = labels[label_id] if labels is not None else label_id
				offset = np.sum(n_objects_test[:i], dtype=np.int32)
				values = correct_predictions[offset+j:offset+n_objects_test[i]:params.DATASET_NUMBER_MATERIALS]
				overall_accuracy.append(values)
				n_is_correct = np.count_nonzero(values)
				n_objects = values.shape[0]
				print(label+":", str(n_is_correct)+"/"+str(n_objects), np.round(np.mean(values), 3))
		print("Overall:", np.round(np.mean(overall_accuracy), 3))

		if params.DATASET_NUMBER_CATEGORIES > 0:
			print("\n*** ACCURACY PER OBJECT ***")
			factor = int(len(labels) / max(params.DATASET_NUMBER_CATEGORIES, 1)) if params.DATASET_IS_SINGLELABEL else 1
			for i in range(params.DATASET_NUMBER_CATEGORIES):
				label = "".join(labels[i*factor].split("_")[:-1]) if labels is not None else i
				offset = np.sum(n_objects_test[0:i], dtype=np.int32)
				values = correct_predictions[offset:offset+n_objects_test[i]]
				n_is_correct = np.count_nonzero(values)
				n_objects = values.shape[0]
				print(label+":", str(n_is_correct)+"/"+str(n_objects), np.round(np.mean(values), 3))

		if params.DATASET_NUMBER_MATERIALS > 0:
			print("\n*** ACCURACY PER MATERIAL ***")
			for i in range(params.DATASET_NUMBER_MATERIALS):
				values = correct_predictions[i::params.DATASET_NUMBER_MATERIALS]
				n_is_correct = np.count_nonzero(values)
				n_objects = values.shape[0]
				print("Material"+str(i)+":", str(n_is_correct)+"/"+str(n_objects),np.round(np.mean(values), 3))
