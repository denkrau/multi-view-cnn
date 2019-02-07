"""
Store global variables
"""

# Datasets
DATASET_PATH = "D:\\Dropbox\\Studium\\Master\\Masterthesis\\Code\\datasets\\modelnet8views"
DATASET_LABELS = ["airplane", "bathtub", "bed", "bench", "bookshelf",
                  "bottle", "bowl", "car"]
IMAGE_SIZE = 224

# Model
LEARNING_RATE = 0.001
N_INPUT = IMAGE_SIZE
N_CLASSES = len(DATASET_LABELS)
N_VIEWS = 12
TRAINING_ITERS = 20
BATCH_SIZE = 10
CKPT_PATH = ".\\checkpoints"
CKPT_FILE = "model.ckpt"
CKPT_OVERWRITE = True
