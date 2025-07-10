import pickle
import sys
import os
import threading

import cv2
import imutils
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_first')
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QInputDialog, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import time
from FaceDetector import faceDetector
from fr_utils import img_to_encoding, load_weights_from_FaceNet
from inception_blocks_v2 import faceRecoModel
import keras
import tensorflow as tf

from parameters import *  # chứa THRESHOLD, ALPHA
import paho.mqtt.client as mqtt

# ============= MQTT connection for smart home door opening ===========

# MQTT_BROKER = "192.168.101.8"
# MQTT_PORT = 1884
# MQTT_USERNAME = "admin"
# MQTT_PASSWORD = "1234"
# MQTT_CLIENT_ID = "face_recognition_app"
# TOPIC_CONTROL = "home/mainDoor/control"
# TOPIC_STATUS = "home/mainDoor/status"
#
# # Callback on_connect: thêm tham số properties để tránh warning
# def on_connect(client, userdata, flags, rc, properties=None):
#     if rc == 0:
#         print("Kết nối MQTT thành công!")
#     else:
#         print(f"Kết nối MQTT thất bại, mã lỗi: {rc}")
#
# client = mqtt.Client(client_id=MQTT_CLIENT_ID)
# client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
# client.on_connect = on_connect
# client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
# client.loop_start()

#
# def lock_door():
#     client.publish(TOPIC_CONTROL, "LOCK")
#     client.publish(TOPIC_STATUS, "LOCK")
#     print("🔒 Gửi lệnh: LOCK")
#
#
#
# def unlock_door():
#     client.publish(TOPIC_CONTROL, "OPEN")
#     client.publish(TOPIC_STATUS, "OPEN")
#     print("🔓 Gửi lệnh: OPEN")
#
#     # Tạo luồng khóa cửa sau 10 giây
#     def delayed_lock():
#         time.sleep(5)
#         lock_door()
#
#     threading.Thread(target=delayed_lock).start()
#


# Hàm verify
def verify(image, identity, database, model):
    encoding = img_to_encoding(image, model, False)
    min_dist = 1000
    for pic in database:
        dist = np.linalg.norm(encoding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' + str(min_dist) + ' ' + str(len(database)))

    if min_dist < THRESHOLD:
        door_open = True
    else:
        door_open = False

    return min_dist, door_open


IMAGE_SIZE = 96  # kích thước ảnh vào model
fd = faceDetector('fd_models/haarcascade_frontalface_default.xml')


def triplet_loss(y_true, y_pred, alpha=ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


best_model_path = ""
if os.path.exists("bestmodel_osht.txt"):
    with open('bestmodel_osht.txt', 'r') as file:
        best_model_path = file.read()

# best_model_path = ""
# if os.path.exists("bestmodel.txt"):
#     with open('bestmodel.txt', 'r') as file:
#         best_model_path = file.read()
# with open("./path_dict_dataset.p", 'rb') as f:
#     paths = pickle.load(f)

# faces = []
# for key in paths.keys():
#     paths[key] = paths[key].replace("\\", "/")
#     faces.append(key)

database_path = "./database"
if not os.path.exists(database_path) or len(os.listdir(database_path)) == 0:
    print("No images found in database!!")
    print("Please add images to database")
    sys.exit()

faces = [name for name in os.listdir(database_path)
         if os.path.isdir(os.path.join(database_path, name))]

if len(faces) == 0:
    print("No images found in database!!")
    print("Please add images to database")
    sys.exit()

if os.path.exists(best_model_path) and best_model_path != "":
    print("Trained model found")
    print("Loading custom trained model...")
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})
else:
    print("Custom trained model not found, Loading original facenet model...")
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)

database = {}
for face in faces:
    database[face] = []

# for face in faces:
#     for img in os.listdir(paths[face]):
#         database[face].append(img_to_encoding(os.path.join(paths[face], img), FRmodel))

for face in faces:
    person_dir = os.path.join(database_path, face)
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        database[face].append(img_to_encoding(img_path, FRmodel))

class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 900, 700)

        self.database = database
        self.FRmodel = FRmodel
        self.fd = fd
        self.secret_code = 111111

        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.mode = None  # 'verify' hoặc 'add'
        self.add_count = 0
        self.new_member_name = None
        self.current_frame = None  # lưu frame hiện tại từ camera

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.video_label = QLabel("")
        self.show_black_image()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.video_label)

        self.result_label = QLabel("Chọn thao tác")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; color: green; font-weight: bold;")
        layout.addWidget(self.result_label)
        
        # Tạo layout hàng ngang để chứa 2 nút chính
        top_button_layout = QHBoxLayout()
        self.btn_verify = QPushButton("Verify Khuôn Mặt")
        self.btn_add = QPushButton("Thêm Thành Viên Mới")

        self.btn_verify.clicked.connect(self.start_verify)
        self.btn_add.clicked.connect(self.start_add_member)


        for btn in [self.btn_verify, self.btn_add]:
            btn.setFixedSize(250, 60)
            btn.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.btn_delete = QPushButton("Xóa Thành Viên")
        self.btn_delete.clicked.connect(self.delete_member)
        self.btn_delete.setFixedSize(250, 60)
        self.btn_delete.setStyleSheet("font-size: 20px; font-weight: bold;")
        top_button_layout.addSpacing(20)
        top_button_layout.addWidget(self.btn_delete)
        top_button_layout.addStretch(1)  # đẩy nút vào giữa
        top_button_layout.addWidget(self.btn_verify)
        top_button_layout.addSpacing(20)  # khoảng cách giữa 2 nút
        top_button_layout.addWidget(self.btn_add)
        top_button_layout.addStretch(1)  # đẩy nút vào giữa

        layout.addLayout(top_button_layout)

        # Các nút còn lại giữ nguyên
        self.btn_exit = QPushButton("Thoát")
        self.btn_exit.clicked.connect(self.exit_mode)
        self.btn_exit.hide()
        self.btn_exit.setFixedSize(250, 60)
        self.btn_exit.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.btn_exit, alignment=Qt.AlignCenter)

        self.btn_do_verify = QPushButton("Verify")
        self.btn_do_verify.clicked.connect(self.do_verify_face)
        self.btn_do_verify.hide()
        self.btn_do_verify.setFixedSize(250, 60)
        self.btn_do_verify.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.btn_do_verify, alignment=Qt.AlignCenter)

        self.btn_capture_face = QPushButton("Thêm Ảnh Khuôn Mặt")
        self.btn_capture_face.clicked.connect(self.capture_face)
        self.btn_capture_face.hide()
        self.btn_capture_face.setFixedSize(250, 60)
        self.btn_capture_face.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.btn_capture_face, alignment=Qt.AlignCenter)



        self.setLayout(layout)

    def start_verify(self):
        self.mode = 'verify'
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        self.timer.start(30)

        self.btn_verify.hide()
        self.btn_add.hide()
        self.btn_delete.hide()
        self.btn_exit.show()
        self.btn_do_verify.show()
        self.btn_capture_face.hide()

        self.video_label.setText("Chế độ Verify. Nhấn 'Verify' để xác nhận khuôn mặt.")


    def do_verify_face(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Không có hình ảnh từ camera!")
            return

        frame = self.current_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = self.fd.detect(gray)

        if len(faceRects) == 0:
            QMessageBox.information(self, "Kết quả", "Không tìm thấy khuôn mặt!")
            return

        (x, y, w, h) = faceRects[0]
        roi = frame[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE))

        identity = None
        min_dist = 1000
        for person in self.database.keys():
            dist, detected = verify(roi_resized, person, self.database[person], self.FRmodel)
            if detected and dist < min_dist:
                min_dist = dist
                identity = person
        if identity:
            self.result_label.setText(f" Nhận diện: {identity} (khoảng cách: {min_dist:.4f})")
            self.result_label.setStyleSheet("font-size: 20px; color: green; font-weight: bold;")
            # unlock_door()
        else:
            self.result_label.setText(" Không nhận diện được khuôn mặt!")
            self.result_label.setStyleSheet("font-size: 20px; color: red; font-weight: bold;")
            # lock_door()

    def start_add_member(self):
        code, ok = QInputDialog.getText(self, "Mã Bảo Mật", "Nhập mã 6 chữ số:")
        if not ok:
            return
        if code.strip() != str(self.secret_code):
            QMessageBox.warning(self, "Lỗi", "Mã không đúng!")
            return

        name, ok2 = QInputDialog.getText(self, "Tên thành viên", "Nhập tên thành viên mới:")
        if not ok2 or not name.strip():
            return

        self.new_member_name = name.strip()
        save_dir = f"./database/{self.new_member_name}"
        os.makedirs(save_dir, exist_ok=True)

        self.mode = 'add'
        self.add_count = 0
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        self.timer.start(30)

        self.btn_verify.hide()
        self.btn_add.hide()
        self.btn_delete.hide()
        self.btn_exit.show()
        self.btn_do_verify.hide()
        self.btn_capture_face.show()

        self.video_label.setText(f"Đã thêm 0 ảnh. Nhấn 'Thêm Ảnh Khuôn Mặt' để chụp.")

    def capture_face(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Không có hình ảnh từ camera!")
            return

        frame = self.current_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = self.fd.detect(gray)

        if len(faceRects) == 0:
            QMessageBox.information(self, "Thông báo", "Không tìm thấy khuôn mặt, vui lòng thử lại!")
            return

        (x, y, w, h) = faceRects[0]
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (96, 96))
        save_path = f"./database/{self.new_member_name}/{self.new_member_name}_{self.add_count}.jpg"
        cv2.imwrite(save_path, roi)
        self.add_count += 1

        self.video_label.setText("Đã thêm 1 ảnh. Đang cập nhật dữ liệu...")

        # Thêm encoding vào database
        self.database[self.new_member_name] = []
        img_path = f"./database/{self.new_member_name}/{self.new_member_name}_0.jpg"
        encoding = img_to_encoding(img_path, self.FRmodel)
        self.database[self.new_member_name].append(encoding)

        QMessageBox.information(self, "Hoàn thành",
                                f"Đã thêm thành viên {self.new_member_name} với 1 ảnh.")
        self.exit_mode()

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return
        frame = imutils.resize(frame, width=800)
        self.current_frame = frame.copy()

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def delete_member(self):
        code, ok = QInputDialog.getText(self, "Mã Bảo Mật", "Nhập mã 6 chữ số:")
        if not ok:
            return
        if code.strip() != str(self.secret_code):
            QMessageBox.warning(self, "Lỗi", "Mã không đúng!")
            return

        members = list(self.database.keys())
        if not members:
            QMessageBox.information(self, "Thông báo", "Không có thành viên nào trong database!")
            return

        name, ok2 = QInputDialog.getItem(self, "Xóa Thành Viên", "Chọn thành viên cần xóa:", members, 0, False)
        if not ok2 or not name:
            return

        # Xác nhận lại trước khi xóa
        confirm = QMessageBox.question(self, "Xác nhận xóa",
                                       f"Bạn có chắc muốn xóa thành viên '{name}' không?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            # Xóa folder trên ổ đĩa
            folder_path = os.path.join(database_path, name)
            if os.path.exists(folder_path):
                import shutil
                shutil.rmtree(folder_path)
            # Xóa khỏi bộ nhớ
            del self.database[name]
            QMessageBox.information(self, "Thành công", f"Đã xóa thành viên '{name}' khỏi database.")
        else:
            QMessageBox.information(self, "Hủy bỏ", "Đã hủy thao tác xóa.")

    def exit_mode(self):
        self.mode = None
        self.new_member_name = None
        self.add_count = 0

        # Dừng camera và timer
        if self.timer.isActive():
            self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # Đặt lại giao diện
        self.btn_verify.show()
        self.btn_add.show()
        self.btn_delete.show()
        self.btn_exit.hide()
        self.btn_do_verify.hide()
        self.btn_capture_face.hide()

        self.video_label.setText(".")
        self.show_black_image()
        self.result_label.setText("Chọn thao tác")
        self.result_label.setStyleSheet("font-size: 20px; color: green; font-weight: bold;")
    def show_black_image(self):
        black_img = np.zeros((600, 800, 3), dtype=np.uint8)  # ảnh đen 800x600
        qt_img = QImage(black_img.data, black_img.shape[1], black_img.shape[0],
                        black_img.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())

    client.loop_stop()
    client.disconnect()