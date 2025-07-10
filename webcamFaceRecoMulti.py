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

from hard_triplet_mining import preprocess_image, collect_image_paths

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
from parameters import *
import pickle
import sys
# =============================================================================
# np.set_printoptions(threshold=np.nan)
# =============================================================================
import keras

best_model_path =""
if(os.path.exists("bestmodel_osht.txt")):
    with open('bestmodel_osht.txt', 'r') as file:
        best_model_path = file.read()
    
with open("./path_dict_dataset.p", 'rb') as f:
    paths = pickle.load(f)
    
faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)
    
if(len(faces) == 0):
    print("No images found in database!!")
    print("Please add images to database")
    sys.exit()

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

if os.path.exists(best_model_path) and best_model_path !="":
    print("Trained model found")
    print("Loading custom trained model...")
    FRmodel = keras.models.load_model(best_model_path,custom_objects={'triplet_loss': triplet_loss})

else:
    print("Custom trained model not found, Loading original facenet model...")
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)



def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model, False)
    min_dist = 1000
    for  pic in database:
        dist = np.linalg.norm(encoding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' +str(min_dist)+ ' ' + str(len(database)))
    
    if min_dist<THRESHOLD:
        door_open = True
    else:
        door_open = False
        
    return min_dist, door_open

#
# database = {}
# for face in faces:
#     database[face] = []
#
# for face in faces:
#     for img in os.listdir(paths[face]):
#         database[face].append(img_to_encoding(os.path.join(paths[face],img), FRmodel))
#
#
# camera = cv2.VideoCapture(0)
# fd = faceDetector('fd_models/haarcascade_frontalface_default.xml')
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec for video
# out = cv2.VideoWriter('output.avi', fourcc, 20, (800, 600) )#Output object
#
# while True:
#     ret, frame = camera.read()
#     frame = imutils.resize(frame, width = 800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     print(frame.shape)
#     faceRects = fd.detect(gray)
#     for (x, y, w, h) in faceRects:
#         roi = frame[y:y+h,x:x+w]
#         roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#         roi = cv2.resize(roi,(IMAGE_SIZE, IMAGE_SIZE))
#         min_dist = 1000
#         identity = ""
#         detected  = False
#
#         for face in range(len(faces)):
#             person = faces[face]
#             dist, detected = verify(roi, person, database[person], FRmodel)
#             if detected == True and dist<min_dist:
#                 min_dist = dist
#                 identity = person
#         if detected == True:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
#         else:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Unknown", (x + (w // 2), y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),lineType=cv2.LINE_AA)
#     cv2.imshow('frame', frame)
#     out.write(frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# verify_result = None  # Lưu kết quả nhận diện
# verify_face_rect = None  # Lưu vị trí khuôn mặt sau khi verify
#
# while True:
#     ret, frame = camera.read()
#     frame = imutils.resize(frame, width=800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faceRects = fd.detect(gray)
#
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('v'):
#         if len(faceRects) > 0:
#             # Chỉ lấy khuôn mặt đầu tiên tại thời điểm nhấn V
#             (x, y, w, h) = faceRects[0]
#             roi = frame[y:y+h, x:x+w]
#             roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#             roi_resized = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE))
#
#             min_dist = 1000
#             identity = "Unknown"
#
#             for person in faces:
#                 dist, detected = verify(roi_resized, person, database[person], FRmodel)
#                 if detected and dist < min_dist:
#                     min_dist = dist
#                     identity = person
#
#             verify_result = identity
#             verify_face_rect = (x, y, w, h)
#
#     # Vẽ khung và tên nếu đã verify
#     if verify_result is not None and verify_face_rect is not None:
#         (x, y, w, h) = verify_face_rect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, verify_result, (x + (w // 2), y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Hiển thị frame
#     cv2.imshow('frame', frame)
#     out.write(frame)
#
#     if key == ord('q'):
#         camera.release()
#         out.release()
#         cv2.destroyAllWindows()
#         break
#
#
#
# verify_result = None  # Lưu kết quả nhận diện
# verify_face_rect = None  # Lưu vị trí khuôn mặt sau khi verify
#
# while True:
#     ret, frame = camera.read()
#     frame = imutils.resize(frame, width=800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faceRects = fd.detect(gray)
#
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('v'):
#         if len(faceRects) > 0:
#             # Chỉ lấy khuôn mặt đầu tiên tại thời điểm nhấn V
#             (x, y, w, h) = faceRects[0]
#             roi = frame[y:y+h, x:x+w]
#             roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#             roi_resized = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE))
#
#             min_dist = 1000
#             identity = "Unknown"
#
#             for person in faces:
#                 dist, detected = verify(roi_resized, person, database[person], FRmodel)
#                 if detected and dist < min_dist:
#                     min_dist = dist
#                     identity = person
#
#             verify_result = identity
#             verify_face_rect = (x, y, w, h)
#
#     # Vẽ khung và tên nếu đã verify
#     if verify_result is not None and verify_face_rect is not None:
#         (x, y, w, h) = verify_face_rect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, verify_result, (x + (w // 2), y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Hiển thị frame
#     cv2.imshow('frame', frame)
#     out.write(frame)
#
#     if key == ord('q'):
#         camera.release()
#         out.release()
#         cv2.destroyAllWindows()
#         break
#
#
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def create_pairs_from_paths_labels(paths, labels, num_same=100, num_diff=100):
    from collections import defaultdict
    label_to_paths = defaultdict(list)
    for path, label in zip(paths, labels):
        label_to_paths[label].append(path)

    label_list = list(label_to_paths.keys())
    pairs = []
    pair_labels = []

    # Positive pairs (same person)
    for _ in range(num_same):
        label = random.choice(label_list)
        if len(label_to_paths[label]) < 2:
            continue
        p1, p2 = random.sample(label_to_paths[label], 2)
        pairs.append((p1, p2))
        pair_labels.append(1)

    # Negative pairs (different people)
    for _ in range(num_diff):
        l1, l2 = random.sample(label_list, 2)
        if len(label_to_paths[l1]) == 0 or len(label_to_paths[l2]) == 0:
            continue
        p1 = random.choice(label_to_paths[l1])
        p2 = random.choice(label_to_paths[l2])
        pairs.append((p1, p2))
        pair_labels.append(0)

    return pairs, pair_labels
# Đã có từ trước:
# - model: mô hình trả về embedding
# - preprocess_image: hàm xử lý ảnh, đầu ra (3, image_size, image_size)

def evaluate_test_set(model, test_paths, test_labels, image_size=96, distance='euclidean', threshold=0.5):
    pairs, labels = create_pairs_from_paths_labels(test_paths, test_labels, num_same=200, num_diff=200)
    preds = []

    for path1, path2 in pairs:
        img1 = preprocess_image(path1, image_size)
        img2 = preprocess_image(path2, image_size)

        emb1 = model.predict(np.expand_dims(img1, axis=0))[0]
        emb2 = model.predict(np.expand_dims(img2, axis=0))[0]

        if distance == 'euclidean':
            dist = np.linalg.norm(emb1 - emb2)
            sim = 1.0 - dist  # similarity = ngược với khoảng cách
        elif distance == 'cosine':
            sim = cosine_similarity([emb1], [emb2])[0][0]
        else:
            raise ValueError("distance phải là 'euclidean' hoặc 'cosine'")

        pred = 1 if sim > threshold else 0
        preds.append(pred)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"[Evaluation on Test Set]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
test_paths, test_labels = collect_image_paths('split_dataset/test')
evaluate_test_set(FRmodel, test_paths, test_labels, image_size=96, distance='euclidean', threshold=0.5)
