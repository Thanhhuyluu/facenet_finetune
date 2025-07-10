# import sys
# from collections import Counter
#
# import itertools
# import random
# import os
# import numpy as np
# from tqdm import tqdm
# import keras
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import random
# from sklearn.model_selection import train_test_split
# import cv2
# from keras import backend as K
#
# import random
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
#
#
# K.set_image_data_format('channels_first')
#
#
#
# sys.path.append(os.path.abspath("..."))
# from parameters import ALPHA, IMAGE_SIZE, LAYERS_TO_FREEZE
# from hard_triplet_mining import collect_image_paths, preprocess_image
# from fr_utils import *
# from inception_blocks_v2 import faceRecoModel
#
# def triplet_loss(y_true, y_pred, alpha=ALPHA):
#     anchor = y_pred[:, 0]
#     positive = y_pred[:, 1]
#     negative = y_pred[:, 2]
#
#     pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
#     neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
#     basic_loss = pos_dist - neg_dist + alpha
#     loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
#     return loss
# best_model_path = None
# if os.path.exists("./bestmodel_osht.txt"):
#     with open('bestmodel_osht.txt', 'r') as file:
#         best_model_path = file.read()
#
# if best_model_path != None and os.path.exists(best_model_path):
#     print("Pre trained model found")
#     FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})
# else:
#     print('Saved model not found, loading untrained FaceNet')
#     FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
#     load_weights_from_FaceNet(FRmodel)
#
# # Freeze layers
# for layer in FRmodel.layers[0: LAYERS_TO_FREEZE]:
#     layer.trainable = False
#
# # Model inputs
# input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
# A = Input(shape=input_shape, name='anchor')
# P = Input(shape=input_shape, name='anchorPositive')
# N = Input(shape=input_shape, name='anchorNegative')
#
# # Get embeddings
# enc_A = FRmodel(A)
# enc_P = FRmodel(P)
# enc_N = FRmodel(N)
#
#
# test_path = "split_dataset/test"
# thresholds = np.arange(0.3, 1.2, 0.05)  # các ngưỡng để kiểm thử
# pairs = []
# labels = []
#
# # ==== Tạo dữ liệu cặp (pair) ====
# folders = os.listdir(test_path)
# folders = [f for f in folders if os.path.isdir(os.path.join(test_path, f))]
# #
# # for i in range(1000):  # tạo 100 cặp ngẫu nhiên
# #     is_same = random.choice([0, 1])
# #
# #     if is_same:
# #         person = random.choice(folders)
# #         imgs = os.listdir(os.path.join(test_path, person))
# #         if len(imgs) < 2:
# #             continue
# #         img1, img2 = random.sample(imgs, 2)
# #         path1 = os.path.join(test_path, person, img1)
# #         path2 = os.path.join(test_path, person, img2)
# #     else:
# #         person1, person2 = random.sample(folders, 2)
# #         img1 = random.choice(os.listdir(os.path.join(test_path, person1)))
# #         img2 = random.choice(os.listdir(os.path.join(test_path, person2)))
# #         path1 = os.path.join(test_path, person1, img1)
# #         path2 = os.path.join(test_path, person2, img2)
# #
# #     pairs.append((path1, path2))
# #     labels.append(is_same)
#
#
# # ==== Lấy embedding ====
# def get_embedding(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#     return img_to_encoding(img, FRmodel, False)
#
# # ========== Tính khoảng cách ==============
#
# #
# # distances = []
# # for path1, path2 in tqdm(pairs):
# #     emb1 = get_embedding(path1)
# #     emb2 = get_embedding(path2)
# #     dist = np.linalg.norm(emb1 - emb2)
# #     distances.append(dist)
# #
# #
# #
# # #
# # # ==== Kiểm thử với nhiều ngưỡng ====
# #
# # all_results = {}
# #
# # for th in thresholds:
# #     preds = [1 if d < th else 0 for d in distances]
# #     acc = accuracy_score(labels, preds)
# #     print(f"Threshold: {th:.2f}, Accuracy: {acc:.4f}")
# #
# #     true_positive_pairs = []
# #     true_negative_pairs = []
# #     false_positive_pairs = []
# #     false_negative_pairs = []
# #
# #     for (pair, label, dist) in zip(pairs, labels, distances):
# #         pred = 1 if dist < th else 0
# #         if pred == 1 and label == 1:
# #             true_positive_pairs.append(pair)
# #         elif pred == 0 and label == 0:
# #             true_negative_pairs.append(pair)
# #         elif pred == 1 and label == 0:
# #             false_positive_pairs.append(pair)
# #         elif pred == 0 and label == 1:
# #             false_negative_pairs.append(pair)
# #
# #     all_results[round(th, 2)] = {
# #         "accuracy": acc,
# #         "true_positive_pairs": true_positive_pairs,
# #         "true_negative_pairs": true_negative_pairs,
# #         "false_positive_pairs": false_positive_pairs,
# #         "false_negative_pairs": false_negative_pairs,
# #     }
# #
# # with open("all_threshold_results.pkl", "wb") as f:
# #     pickle.dump(all_results, f)
# #
# # print("✅ Đã lưu kết quả tại nhiều threshold vào all_threshold_results.pkl")
# #
#
#
#
# import pickle
# import cv2
# import matplotlib.pyplot as plt
#
# #========= cho chọn threshold để xem kế quả ========================
# with open("all_threshold_results.pkl", "rb") as f:
#     all_results = pickle.load(f)
#
# # Hiển thị danh sách threshold
# thresholds_available = list(all_results.keys())
# print("Các threshold có sẵn:", thresholds_available)
#
# def show_pairs(pair_list, title, max_pairs=5):
#     plt.figure(figsize=(10, 4 * max_pairs))
#     for idx, (p1, p2) in enumerate(pair_list[:max_pairs]):
#         img1 = cv2.imread(p1)
#         img2 = cv2.imread(p2)
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#
#         plt.subplot(max_pairs, 2, 2 * idx + 1)
#         plt.imshow(img1)
#         plt.title(f"{title} - Image 1")
#         plt.axis('off')
#
#         plt.subplot(max_pairs, 2, 2 * idx + 2)
#         plt.imshow(img2)
#         plt.title(f"{title} - Image 2")
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# while True:
#     # Nhập threshold
#     selected = input("\nNhập threshold bạn muốn xem (hoặc -1 để thoát): ").strip()
#     if selected == "-1":
#         print("👋 Kết thúc.")
#         break
#
#     try:
#         selected_th = float(selected)
#     except ValueError:
#         print("❌ Vui lòng nhập một số thực hoặc -1.")
#         continue
#
#     if selected_th not in all_results:
#         print("❌ Threshold không hợp lệ.")
#         continue
#
#     data = all_results[selected_th]
#
#     # Hiển thị thông tin tổng quan
#     print(f"\n📊 Kết quả tại threshold {selected_th}:")
#     print(f"Accuracy: {data['accuracy']:.4f}")
#     print(f"True Positives: {len(data['true_positive_pairs'])}")
#     print(f"False Positives: {len(data['false_positive_pairs'])}")
#     print(f"True Negatives: {len(data['true_negative_pairs'])}")
#     print(f"False Negatives: {len(data['false_negative_pairs'])}")
#
#     # Chọn loại để hiển thị
#     choice = input("\nChọn loại để hiển thị (tp, tn, fp, fn, hoặc -1 để quay lại): ").strip().lower()
#
#     if choice == "-1":
#         continue
#
#     mapping = {
#         "tp": "true_positive_pairs",
#         "tn": "true_negative_pairs",
#         "fp": "false_positive_pairs",
#         "fn": "false_negative_pairs",
#     }
#
#     if choice not in mapping:
#         print("❌ Loại không hợp lệ.")
#         continue
#
#     pair_list = data[mapping[choice]]
#     show_pairs(pair_list, choice.upper())
#
#
#
# # ====== vẽ bảng===========
#
#
# import pickle
#
# import pandas as pd
#
# # Load kết quả đã lưu
# with open("all_threshold_results.pkl", "rb") as f:
#     all_results = pickle.load(f)
#
# # Tạo bảng thống kê
# summary = {
#     "Threshold": [],
#     "Accuracy": [],
#     "True Positives": [],
#     "False Positives": [],
#     "True Negatives": [],
#     "False Negatives": [],
# }
#
# for th, data in all_results.items():
#     summary["Threshold"].append(th)
#     summary["Accuracy"].append(data["accuracy"])
#     summary["True Positives"].append(len(data["true_positive_pairs"]))
#     summary["False Positives"].append(len(data["false_positive_pairs"]))
#     summary["True Negatives"].append(len(data["true_negative_pairs"]))
#     summary["False Negatives"].append(len(data["false_negative_pairs"]))
#
# # Chuyển thành DataFrame để dễ hiển thị
# df = pd.DataFrame(summary)
# df = df.sort_values(by="Threshold")
#
# # Hiển thị bảng
# print("\n📊 Tổng hợp kết quả theo từng threshold:")
# print(df.to_string(index=False))
#
# # Vẽ biểu đồ nếu muốn
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 6))
# plt.plot(df["Threshold"], df["True Positives"], label="TP", marker='o')
# plt.plot(df["Threshold"], df["False Positives"], label="FP", marker='o')
# plt.plot(df["Threshold"], df["True Negatives"], label="TN", marker='o')
# plt.plot(df["Threshold"], df["False Negatives"], label="FN", marker='o')
# plt.xlabel("Threshold")
# plt.ylabel("Số lượng cặp")
# plt.title("Biến thiên TP/FP/TN/FN theo threshold")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import cv2
import numpy as np

from hard_triplet_mining import collect_image_paths

test_paths, test_labels = collect_image_paths('split_dataset/test')
def get_embeddings(model, image_paths, image_size):
    embeddings = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = img[..., ::-1]  # BGR to RGB
        img = img / 255.
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = np.expand_dims(img, axis=0)
        embedding = model.predict(img)
        embeddings.append(embedding[0])
    return np.array(embeddings)
X_test = get_embeddings(FRmodel, test_paths, IMAGE_SIZE)
y_test = np.array(test_labels)
