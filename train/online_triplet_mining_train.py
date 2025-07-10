import csv

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.callbacks import Callback
import numpy as np
from hard_triplet_mining import hard_triplet_batch_generator, semi_hard_triplet_batch_generator_no_loop, \
    online_triplet_generator, collect_image_paths

K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import imutils
from FaceDetector import *
from keras.models import model_from_json
import keras
from generator_utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time
from parameters import *
from keras.optimizers import Adam



def triplet_loss(y_true, y_pred, alpha=ALPHA):
    anchor = y_pred[:, 0]
    positive = y_pred[:, 1]
    negative = y_pred[:, 2]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


best_model_path = None
if os.path.exists("../bestmodel_osht.txt"):
    with open('../bestmodel_osht.txt', 'r') as file:
        best_model_path = file.read()

if best_model_path != None and os.path.exists(best_model_path):
    ("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})
else:
    print('Saved model not found, loading untrained FaceNet')
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)

for layer in FRmodel.layers[0: LAYERS_TO_FREEZE]:
    layer.trainable = False

input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
A = Input(shape=input_shape, name='anchor')
P = Input(shape=input_shape, name='anchorPositive')
N = Input(shape=input_shape, name='anchorNegative')

enc_A = FRmodel(A)
enc_P = FRmodel(P)
enc_N = FRmodel(N)
graph = tf.get_default_graph()
# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
STAMP = 'facenet_%d' % (len(paths))
checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

from keras.callbacks import CSVLogger

csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'), append=True)

bst_model_path = checkpoint_dir + STAMP + '.h5'
tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

# Model
merged_output = Lambda(lambda x: K.stack(x, axis=1))([enc_A, enc_P, enc_N])
tripletModel = Model(inputs=[A, P, N], outputs=merged_output)

tripletModel.compile(optimizer = 'adam', loss = triplet_loss)

class TripletStatsCallback(Callback):
    def __init__(self, filename="triplet_stats.csv"):
        super().__init__()
        self.filename = filename

        # Nếu file chưa tồn tại, tạo và ghi header
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Avg_Triplets_Train", "Avg_Triplets_Val"])

    def on_epoch_end(self, epoch, logs=None):
        if triplet_counts_per_batch_train:
            avg_train = np.mean(triplet_counts_per_batch_train)
        else:
            avg_train = 0.0

        if triplet_counts_per_batch_val:
            avg_val = np.mean(triplet_counts_per_batch_val)
        else:
            avg_val = 0.0

        print(f"[Epoch {epoch+1}] Avg semi-hard triplets - Train: {avg_train:.2f}, Val: {avg_val:.2f}")

        # Ghi vào file CSV
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train, avg_val])

        # Reset lại danh sách triplet count mỗi epoch
        triplet_counts_per_batch_train.clear()
        triplet_counts_per_batch_val.clear()

train_paths, train_labels = collect_image_paths('../split_dataset/train')
val_paths, val_labels = collect_image_paths('../split_dataset/val')
# test_paths, test_labels = collect_image_paths('split_dataset/test')
steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(val_paths) // BATCH_SIZE
triplet_counts_per_batch_train = []
triplet_counts_per_batch_val = []

train_gen = online_triplet_generator(
    model=FRmodel,
    image_paths=train_paths,
    labels=train_labels,
    graph=graph,
    type="train",
    triplets_count= triplet_counts_per_batch_train,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    alpha=ALPHA

)

val_gen = online_triplet_generator(
    model=FRmodel,
    image_paths=val_paths,
    labels=val_labels,
    graph=graph,
    type="val",
    triplets_count=triplet_counts_per_batch_val,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    alpha=ALPHA
)



history = tripletModel.fit_generator(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs= 10,
    callbacks=[
        ModelCheckpoint("facenet_online_osht_aug.h5", monitor='val_loss', save_best_only=True),
    TripletStatsCallback()
    ]
)
# Lưu loss vào file (có thể là file .txt hoặc pickle)
with open('Evaluate/history_osht_aug.p', 'wb') as f:
    pickle.dump(history.history, f)

FRmodel.save(bst_model_path)
with open('bestmodel_osht_aug.txt', 'w') as file:
    file.write(bst_model_path)
