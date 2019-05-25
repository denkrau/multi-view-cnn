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
import sklearn.metrics as metrics

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

	#split correct_predictions in lists of if classification is correct and classified label id
	is_correct = correct_predictions[:,0]
	correct_label_ids = correct_predictions[:,1]

	if params.DATASET_IS_SINGLELABEL:
		#if write argument is given create file and write results
		if arg_write:
			path = os.path.join(params.RESULTS_PATH, "models", os.path.basename(params.CKPT_PATH))
			f = open(os.path.join(path, "stats.txt"), "w")
			f.write("Overall Accuracy: {:.3f}\n".format(np.mean(is_correct)))

		print("Overall Accuracy:", np.round(np.mean(is_correct), 3))

		if params.DATASET_NUMBER_CATEGORIES > 0 and params.DATASET_NUMBER_MATERIALS > 0:
			if arg_write:
				f.write("\nAccuracy per Label:\n")

			print("*** ACCURACY PER LABEL ***")
			for i in range(params.DATASET_NUMBER_CATEGORIES):
				for j in range(params.DATASET_NUMBER_MATERIALS):
					label_id = i * params.DATASET_NUMBER_MATERIALS + j
					label = labels[label_id] if labels is not None else label_id
					offset = np.sum(n_objects_test[:i], dtype=np.int32)
					values = is_correct[offset+j:offset+n_objects_test[i]:params.DATASET_NUMBER_MATERIALS]
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
				values = is_correct[offset:offset+n_objects_test[i]]
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
				values = is_correct[i::params.DATASET_NUMBER_MATERIALS]
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
			colors_bgr = [(0,0,1.), (0,1.,0)]
			group_metrics = []
			#check for every material but skip blank one
			for i in range(1, params.DATASET_NUMBER_MATERIALS):
				matches = []
				#get all views and related group ids of certain material label
				for views, idx in zip(x_test[i::params.DATASET_NUMBER_MATERIALS], group_ids[i::params.DATASET_NUMBER_MATERIALS]):
					#get view ids of top group
					idx_top_views = np.argwhere(idx == np.amax(idx)).flatten().tolist()
					#check for every view in top group if view contains material
					for j in idx_top_views:
						match = np.zeros(views[0].shape[0:2])
						for c in colors_bgr:
							match = np.logical_or(match, cv2.inRange(views[j], c, c))
						matches.append(np.any(match))
				matches_avg = np.mean(matches)
				#keep track of group metric for each material
				group_metrics.append(np.mean(matches))

				print("Material{}: {:.3f}".format(i, matches_avg))
				if arg_write:
					f.write("{:.3f}\n".format(matches_avg))
			print("Overall: {:.3f}".format(np.mean(group_metrics)))
			if arg_write:
				f.write("{:.3f}\n".format(np.mean(group_metrics)))

		#get several metrics for model evaluation
		#get label id from one hot encoded labels
		y_test_ids = np.argmax(y_test, axis=1)

		#classification report contains precision, recall and f1-score for each category
		classification_report = metrics.classification_report(y_test_ids, correct_label_ids, target_names=labels)
		print("\n*** CLASSIFICATION REPORT ***")
		print(classification_report)

		#confusion score counts right and wrong prediction per category
		confusion_score = metrics.confusion_matrix(y_test_ids, correct_label_ids).tolist()
		#make the output pretty
		for i, line in enumerate(confusion_score):
			line.insert(0, labels[i])
		confusion_score.insert(0, [" ", *labels])

		print("\n*** CONFUSION SCORE ***")
		print(np.array(confusion_score))

		#write metrics to disk
		if arg_write:
			f.write("\nClassification Report:\n")
			f.write(classification_report)
			f.write("\nConfusion Score:\n")
			for line in confusion_score:
				for l in line:
					f.write("{:<10}".format(l))
				f.write("\n")

		if arg_write:
			f.close()

		#save filenames of wrong predictions to disk
		if arg_write:
			path = os.path.join(params.RESULTS_PATH, "models", os.path.basename(params.CKPT_PATH))
			is_correct, pred_id = np.split(correct_predictions, indices_or_sections=2, axis=1)
			with open(os.path.join(path, "misclassifications.txt"), "w") as f:
				misclassifications = [[n, p] for n, c, p in zip(data.get_filenames(), is_correct, pred_id) if not c]
				for n, p in misclassifications:
					f.write("{} {}\n".format(n, labels[p[0]]))