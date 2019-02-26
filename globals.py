"""
Store global variables
"""

# Datasets
DATASET_PATH = ".\\datasets\\modelnet8views"
DATASET_LABELS = ["airplane", "bathtub", "bed", "bench", "bookshelf",
                  "bottle", "bowl", "car"]
IMAGE_SIZE = 224

# Model
#LEARNING_RATE_TYPES = ["Fixed", "exp", "cyclic", "cos_cyclic"]
LEARNING_RATE_TYPE = 0
LEARNING_RATE = 0.0001
FIND_LEARNING_RATE_MIN = 0.00001
FIND_LEARNING_RATE_GROWTH = 1.1
N_INPUT = IMAGE_SIZE
N_CLASSES = len(DATASET_LABELS)
N_VIEWS = 12
TRAINING_EPOCHS = 10
BATCH_SIZE_SINGLE = 128
BATCH_SIZE_MULTI = 8
CKPT_PATH = ".\\checkpoints"
CKPT_FILE = "model.ckpt"
CKPT_OVERWRITE = True

# Summary
USE_SUMMARY = False
SUMMARY_PATH = "./Output"
