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
from matplotlib2tikz import save as save_tikz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
path_predictions = os.path.join(params.RESULTS_PATH, "Predictions")
matplotlib.rcParams["savefig.directory"] = path_predictions
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
	#get name of used model/checkpoint
	if ckpt_file is None:
		model_name = os.path.basename(params.CKPT_PATH)
	else:
		model_name = os.path.basename(os.path.dirname(ckpt_file))

	#join fill string if given
	if fill:
		model_name = "_".join([model_name, fill])

	path = os.path.join(path_predictions, model_name)

	#if directory does not exist create it
	if not os.path.isdir(path):
		os.makedirs(path)

	return path

if __name__ == "__main__":
	#first get command line arguments
	args = sys.argv[1:]
	unixOptions = "sgfov:c:w"
	gnuOptions = ["saliency", "groups", "features", "one", "views=", "ckpt=", "write"]
	arg_multi = True
	arg_groups = False
	arg_saliency = False
	arg_views = params.N_VIEWS
	arg_ckpt = None
	arg_features = False
	arg_write = False
	rows_in_plot = 2
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-o", "--one"):
			arg_multi = False
		elif arg in ("-g", "--groups"):
			arg_groups = True
		elif arg in ("-v", "--views"):
			arg_views = int(val)
		elif arg in ("-c", "--ckpt"):
			arg_ckpt = val
		elif arg in ("-f", "--features"):
			arg_features = True
		elif arg in ("-s", "--saliency"):
			arg_saliency = True
		elif arg in ("-w", "--write"):
			arg_write = True

	#initialize objects for further use
	model = model.Model()
	data = data.Data()
	labels = data.load_labels()

	images = []
	if arg_multi is False:
		for val in values:
			img = cv2.imread(val, 1 % params.IMAGE_CHANNELS)
			if img.dtype == np.uint8:
				img = img / 255.0
			if img.shape[0] != params.IMAGE_SIZE and img.shape[1] != params.IMAGE_SIZE:
				img = cv2.resize(img, (params.IMAGE_SIZE, params.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
			images.append(img)
		images = np.reshape(img, [-1, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
	else:
		# airplane_0627_001.jpg
		# Format has to be name_id.jpg
		for val in values:
			head, tail = os.path.split(val)
			name, ext = os.path.splitext(tail)
			name = name.split("_")
			for i in range(arg_views):
				path = os.path.join(head, "_".join([*name[:-1], str(i).zfill(3)])) + ext
				img = cv2.imread(path, 1 % params.IMAGE_CHANNELS)
				if img.dtype == np.uint8:
					img = img / 255.0
				if img.shape[0] != params.IMAGE_SIZE and img.shape[1] != params.IMAGE_SIZE:
					img = cv2.resize(img, (params.IMAGE_SIZE, params.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
				images.append(img)
		images = np.reshape(images, [-1, arg_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])

	#manipulate view filename to get object name for path creation
	if arg_write:
		values = ["_".join(os.path.splitext(os.path.basename(v))[0].split("_")[:-1]) for v in values]
		plot_path = get_save_path(ckpt_file=arg_ckpt)

	#predictions = classifications, saliencies, view_scores, group_ids, group_weights, correct_predictions, activations_convs
	predictions = model.predict(images, ckpt_file=arg_ckpt)

	for classification, saliency, scores, group_ids, group_weights, is_correct, activation_convs, views, obj_name in zip(*predictions, images, values):
		classification = sorted(zip(labels if labels is not None else range(len(classification)), classification), key=lambda x: (x[1]), reverse=True)
		if arg_write:
			f = open(os.path.join(plot_path, "_".join([obj_name, "prediction.txt"])), "w")
			f.write("Classification:\n")
		print("*** CLASSIFICATION ***")
		for i in classification[:5]:
			print(str(i[0])+":", i[1])
			if arg_write:
				f.write("{}: {}\n".format(i[0], i[1]))
		if arg_write:
			f.close()

		if scores.size and group_ids.size and group_weights.size:
			print("\n*** GROUPING ***")
			print("View", *np.arange(arg_views), sep="\t")
			print("=====================================================================================================")
			print("Score", *np.round(scores, 3), sep="\t")
			print("Group", *group_ids, sep="\t")

			print("\nGroup", *np.arange(arg_views), sep="\t")
			print("=====================================================================================================")
			print("Weight", *np.around(group_weights, 3), "\n", sep="\t")

			#plots group distribution of each view
			if arg_groups or arg_write:
				groups = sorted(zip(views, scores, group_ids, saliency), key=lambda x: (x[2], x[1]))
				_, idx = np.unique([i[2] for i in groups], return_index=True)
				fig, ax = plt.subplots(rows_in_plot, int(arg_views/rows_in_plot), figsize=(6.2, 3.2), dpi=300)
				ax = ax.reshape(-1)
				for i in range(arg_views):
					if i in idx:
						ax[i].text(0.02, 1.1, "G{}: {:.3f}".format(np.argwhere(idx==i)[0][0]+1 ,group_weights[groups[i][2]]), size=12, ha="left", transform=ax[i].transAxes)
						#ax[i].text(0.5, 1.1, str(round(group_weights[groups[i][2]], 3))+"~"+str(round(group_weights[groups[i][2]]/np.max(group_weights), 3)), size=12, ha="center", transform=ax[i].transAxes)
					ax[i].text(0.5,-0.19, round(groups[i][1], 3), size=12, ha="center", transform=ax[i].transAxes)
					ax[i].xaxis.set_major_locator(plt.NullLocator())
					ax[i].yaxis.set_major_locator(plt.NullLocator())
					ax[i].imshow(groups[i][0][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)
				plt.tight_layout()
				if arg_write:
					plt.savefig(os.path.join(plot_path, "_".join([obj_name, "grouping.png"])))
					save_tikz(os.path.join(plot_path, "_".join([obj_name, "grouping.tikz"])), figureheight="\\figureheight", figurewidth="\\figurewidth")
				if not arg_groups:
					plt.close()

		#plots saliency maps of each view
		if arg_saliency or arg_write:
			if scores.size and group_ids.size and group_weights.size:
				groups = sorted(zip(views, scores, group_ids, saliency), key=lambda x: (x[2], x[1]))
				saliency = [s[3] for s in groups]
			fig, ax = plt.subplots(rows_in_plot, int(arg_views/rows_in_plot), figsize=(6.2, 2), dpi=300)
			plt.subplots_adjust()
			ax = ax.reshape(-1)
			for i in range(arg_views):
				ax[i].xaxis.set_major_locator(plt.NullLocator())
				ax[i].yaxis.set_major_locator(plt.NullLocator())
				with np.errstate(divide='ignore', invalid='ignore'):
					#normalized representation
					saliency[i] = np.nan_to_num((saliency[i] - np.min(saliency[i])) / (np.max(saliency[i]) - np.min(saliency[i])))
					#dark color representation
					#saliency[i] = np.where(saliency[i] > 0, saliency[i]/np.max(saliency[i]), 0.0)
				ax[i].imshow(saliency[i][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)
			plt.tight_layout()
			if arg_write:
				plt.savefig(os.path.join(plot_path, "_".join([obj_name, "saliency.png"])))
				save_tikz(os.path.join(plot_path, "_".join([obj_name, "saliency.tikz"])), figureheight="\\figureheight", figurewidth="\\figurewidth")
			if not arg_saliency:
				plt.close()

		#plots feature maps of each view
		if arg_features or arg_write:
			kernel_size = activation_convs.shape[-2]
			n_filter = activation_convs.shape[-1]
			grid_size = np.ceil(np.sqrt(n_filter)).astype(int)
			for v in range(params.N_VIEWS):
				fig, axes = plt.subplots(grid_size, grid_size, figsize=(6.2,8), dpi=300)
				fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.1)
				axes = axes.reshape([-1])
				for i, ax in enumerate(axes):
					if i < n_filter:
						img = activation_convs[v,:,:,:,i]
						with np.errstate(divide='ignore', invalid='ignore'):
							img = np.nan_to_num((img - np.min(img)) / (np.max(img) - np.min(img)))
						#img = gaussian_filter(img, sigma=2.0)
					else:
						img = np.full([kernel_size, kernel_size], 1.0)
					ax.axis("off")
					ax.imshow(img.reshape(kernel_size, kernel_size), cmap="gray", vmin=0, vmax=1)
				if arg_write:
					plt.savefig(os.path.join(plot_path, "_".join([obj_name, "activations"+str(v).zfill(2)+".png"])))
					save_tikz(os.path.join(plot_path, "_".join([obj_name, "activations"+str(v).zfill(2)+".tikz"])), figureheight="\\figureheight", figurewidth="\\figurewidth")
				if not arg_features:
					plt.close()

	#if a plot exists show it
	if arg_multi or	arg_groups or arg_saliency or arg_features:
		plt.show()