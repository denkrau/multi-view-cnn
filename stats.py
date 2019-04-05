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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
	args = sys.argv[1:]
	unixOptions = "sw"
	gnuOptions = ["single", "write"]
	arg_multi = True
	arg_write = False
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-s", "--single"):
			arg_multi = False
		if arg in ("-w", "--write"):
			arg_write = True

	#initialize objects for further use
	model = model.Model()
	data = data.Data()

	x_train, y_train, x_test, y_test = data.get_dataset(params.DATASET_PATH, train=False, one_hot=True)
	n_objects_train, n_objects_test = data.get_amount_objects_per_category()

	if arg_multi:
		x_train, y_train, x_test, y_test = data.single_to_multi_view(x_train, y_train, x_test, y_test, params.N_VIEWS)
	labels = data.load_labels()

	_, _, _, group_ids, _, correct_predictions, _ = model.predict(x_test, y_test, get_saliency=False, get_activations=False)

	if params.DATASET_IS_SINGLELABEL:
		#if write argument is given create file and write results
		if arg_write:
			path = os.path.join(params.RESULTS_PATH, "models", os.path.basename(params.CKPT_PATH))
			f = open(os.path.join(path, "stats.txt"), "w")
			f.write("Overall Accuracy: {:.3f}\n".format(np.mean(correct_predictions)))

		print("Overall Accuracy:", np.round(np.mean(correct_predictions), 3))

		if params.DATASET_NUMBER_CATEGORIES > 0 and params.DATASET_NUMBER_MATERIALS > 0:
			if arg_write:
				f.write("\nAccuracy per Label:\n")

			print("*** ACCURACY PER LABEL ***")
			for i in range(params.DATASET_NUMBER_CATEGORIES):
				for j in range(params.DATASET_NUMBER_MATERIALS):
					label_id = i * params.DATASET_NUMBER_MATERIALS + j
					label = labels[label_id] if labels is not None else label_id
					offset = np.sum(n_objects_test[:i], dtype=np.int32)
					values = correct_predictions[offset+j:offset+n_objects_test[i]:params.DATASET_NUMBER_MATERIALS]
					n_is_correct = np.count_nonzero(values)
					n_objects = values.shape[0]
					print(label+":", str(n_is_correct)+"/"+str(n_objects), np.round(np.mean(values), 3))
					if arg_write:
						f.write("{}: {}/{} {:.3f}\n".format(label, n_is_correct, n_objects, np.mean(values)))

		if params.DATASET_NUMBER_CATEGORIES > 0:
			if arg_write:
				f.write("\nAccuracy per Object:\n")
			print("\n*** ACCURACY PER OBJECT ***")
			factor = int(len(labels) / max(params.DATASET_NUMBER_CATEGORIES, 1)) if params.DATASET_IS_SINGLELABEL else 1
			for i in range(params.DATASET_NUMBER_CATEGORIES):
				#remove material id from label if necessary
				if params.DATASET_NUMBER_MATERIALS > 0:
					label = "_".join(labels[i*factor].split("_")[:-1]) if labels is not None else i
				else:
					label = labels[i*factor] if labels is not None else i
				offset = np.sum(n_objects_test[0:i], dtype=np.int32)
				values = correct_predictions[offset:offset+n_objects_test[i]]
				n_is_correct = np.count_nonzero(values)
				n_objects = values.shape[0]
				print(label+":", str(n_is_correct)+"/"+str(n_objects), np.round(np.mean(values), 3))
				if arg_write:
					f.write("{}: {}/{} {:.3f}\n".format(label, n_is_correct, n_objects, np.mean(values)))

		if params.DATASET_NUMBER_MATERIALS > 0:
			if arg_write:
				f.write("\nAccuracy per Label:\n")
			print("\n*** ACCURACY PER MATERIAL ***")
			for i in range(params.DATASET_NUMBER_MATERIALS):
				values = correct_predictions[i::params.DATASET_NUMBER_MATERIALS]
				n_is_correct = np.count_nonzero(values)
				n_objects = values.shape[0]
				print("Material"+str(i)+":", str(n_is_correct)+"/"+str(n_objects),np.round(np.mean(values), 3))
				if arg_write:
					f.write("Material{}: {}/{} {:.3f}\n".format(i, n_is_correct, n_objects, np.mean(values)))
		
		if params.DATASET_NUMBER_MATERIALS > 0:
			if arg_write:
				f.write("\nGroup Metric:\n")
			print("\n*** GROUP METRIC ***")
			#color space is BGR and values are in range [0,1]
			color_bgr = (0,0,1.)
			matches = []
			#get all views and related group ids of certain material label
			for views, idx in zip(x_test[2::params.DATASET_NUMBER_MATERIALS], group_ids[2::params.DATASET_NUMBER_MATERIALS]):
				#get view ids of top group
				idx_top_views = np.argwhere(idx == np.amax(idx)).flatten().tolist()
				#check for every view in top group if view contains material
				for i in idx_top_views:
					match = cv2.inRange(views[i], color_bgr, color_bgr)
					matches.append(np.any(match))
			print(np.round(np.mean(matches), 3))
			if arg_write:
				f.write("{:.3f}\n".format(np.mean(matches)))

		if arg_write:
			f.close()




