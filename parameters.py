import os

if os.path.exists("./cropped")==False:
    os.makedirs("./cropped")

ALPHA = 0.5
THRESHOLD = 0.37
IMAGE_SIZE= 96
LAYERS_TO_FREEZE= 50
NUM_EPOCHS= 100
STEPS_PER_EPOCH= 300
STEPS_PER_EPOCH_VAL = 62
BATCH_SIZE= 128