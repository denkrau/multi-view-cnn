"""
Stores global parameters
"""
import os

# Helper functions
def get_number_labels():
	if DATASET_IS_SINGLELABEL:
		n_classes = DATASET_NUMBER_CATEGORIES * DATASET_NUMBER_MATERIALS \
			if DATASET_NUMBER_CATEGORIES > 0 and DATASET_NUMBER_MATERIALS > 0 \
			else max(DATASET_NUMBER_CATEGORIES, DATASET_NUMBER_MATERIALS)
	else:
		n_classes = DATASET_NUMBER_CATEGORIES + DATASET_NUMBER_MATERIALS

	return n_classes

# Datasets
DATASET_PATH = os.path.join("datasets", "modelnet1-6")
DATASET_FORMATS = [".jpg", ".jpeg", ".png"]
DATASET_NUMBER_CATEGORIES = 0
DATASET_NUMBER_MATERIALS = 6
DATASET_IS_SINGLELABEL = True
DATASET_LABELS_FILE = os.path.join(DATASET_PATH, "labels.txt")
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# Model
#LEARNING_RATE_TYPES = ["Fixed", "exp", "cyclic", "cos_cyclic"]
LEARNING_RATE_TYPE = 0
LEARNING_RATE = 0.0001
N_INPUT = IMAGE_SIZE
N_VIEWS = 12
N_CLASSES = get_number_labels()
TRAINING_EPOCHS = 20
BATCH_SIZE_SINGLE = 128
BATCH_SIZE_MULTI = 8
DROPOUT_PROB = 0.5
CKPT_PATH = os.path.join("checkpoints", "mn-sl-0-6-20")
CKPT_FILE = "model.ckpt"
CKPT_OVERWRITE = True

#Finding learning rate
FIND_LEARNING_RATE_MIN = 0.00001
FIND_LEARNING_RATE_GROWTH = 1.1

# Summary
USE_SUMMARY = True
SUMMARY_PATH = os.path.join("summary", os.path.basename(CKPT_PATH))
USE_PYPLOT = True
RESULTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "Results")