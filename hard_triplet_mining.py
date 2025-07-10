


from parameters import  *
from fr_utils import *
import os
import numpy as np
from tqdm import tqdm
import keras
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.model_selection import train_test_split
import cv2
from keras import backend as K
import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split
K.set_image_data_format('channels_first')


def collect_image_paths(data_dir):
    image_paths = []
    labels = []
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir): continue
        for fname in os.listdir(person_dir):
            if fname.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(person_dir, fname))
                labels.append(person)
    return image_paths, labels


def compute_all_embeddings(image_paths, model):
    embeddings = []
    for path in tqdm(image_paths, desc='Computing embeddings'):
        emb = img_to_encoding(path, model, path=True)
        embeddings.append(emb[0])  # emb is shape (1, 128)
    return np.array(embeddings)


def preprocess_image(img_path, image_size=96):
    """
    Đọc và tiền xử lý ảnh theo chuẩn của FaceNet: resize, chuyển RGB, chuẩn hóa và chuyển channels_first
    """
    img = cv2.imread(img_path)  # BGR
    if img is None:
        raise ValueError(f"Cannot read image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))  # Resize ảnh
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)  # [C, H, W]
    return img

def create_triplets(image_paths, labels, embeddings, alpha=ALPHA):
    triplets = []
    label_to_indices = {}
    label_list = list(set(labels))

    # Tạo từ điển ánh xạ nhãn -> index
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    for label in tqdm(label_list, desc='Creating semi-hard triplets'):
        pos_indices = label_to_indices[label]
        if len(pos_indices) < 2:
            continue

        # Duyệt qua từng anchor-positive pair
        for i in range(len(pos_indices)):
            for j in range(i + 1, len(pos_indices)):
                anchor_idx = pos_indices[i]
                pos_idx = pos_indices[j]

                anchor_emb = embeddings[anchor_idx]
                pos_emb = embeddings[pos_idx]

                d_ap = np.sum(np.square(anchor_emb - pos_emb))

                # So sánh với tất cả negative embeddings
                for neg_idx in range(len(embeddings)):
                    if labels[neg_idx] == label:
                        continue  # Không dùng cùng class

                    neg_emb = embeddings[neg_idx]
                    d_an = np.sum(np.square(anchor_emb - neg_emb))

                    # Semi-hard điều kiện: d_ap < d_an < d_ap + alpha
                    if d_ap < d_an < d_ap + alpha:
                        triplets.append((
                            image_paths[anchor_idx],
                            image_paths[pos_idx],
                            image_paths[neg_idx]
                        ))
                        break  # Chỉ cần 1 negative phù hợp

    return triplets



def show_hard_triplets(triplets, count=5):
    """
    Hiển thị một số triplet: (anchor, positive, negative)
    """
    selected_triplets = random.sample(triplets, count)

    for i, (anchor_path, positive_path, negative_path) in enumerate(selected_triplets):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        for ax, path, title in zip(axs,
                                   [anchor_path, positive_path, negative_path],
                                   ['Anchor', 'Positive', 'Negative']):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f'Triplet {i + 1}')
        plt.tight_layout()
        plt.show()


def hard_triplet_batch_generator(triplets, batch_size, image_size):
    """
    Generator tạo batch từ danh sách triplets (anchor_path, positive_path, negative_path).
    Trả về dictionary {'anchor': ..., 'anchorPositive': ..., 'anchorNegative': ...} và dummy y.
    """
    num_triplets = len(triplets)

    while True:
        np.random.shuffle(triplets)
        for i in range(0, num_triplets, batch_size):
            batch_triplets = triplets[i: i + batch_size]

            anchors = np.zeros((len(batch_triplets), 3, image_size, image_size))
            positives = np.zeros((len(batch_triplets), 3, image_size, image_size))
            negatives = np.zeros((len(batch_triplets), 3, image_size, image_size))

            for j, (anchor_path, pos_path, neg_path) in enumerate(batch_triplets):
                anchors[j] = preprocess_image(anchor_path, image_size)
                positives[j] = preprocess_image(pos_path, image_size)
                negatives[j] = preprocess_image(neg_path, image_size)

            x_data = {
                'anchor': anchors,
                'anchorPositive': positives,
                'anchorNegative': negatives
            }

            y_dummy = np.zeros((len(batch_triplets), 3, IMAGE_SIZE))
            yield x_data, y_dummy

def semi_hard_triplet_batch_generator_no_loop(triplets, batch_size, image_size = IMAGE_SIZE):
    """
    Generator tạo batch từ danh sách triplets (anchor_path, positive_path, negative_path).
    Chạy một vòng qua toàn bộ triplets (không lặp vô hạn).
    """
    num_triplets = len(triplets)
    np.random.shuffle(triplets)  # Shuffle trước 1 lần

    for i in range(0, num_triplets, batch_size):
        batch_triplets = triplets[i: i + batch_size]

        anchors = np.zeros((len(batch_triplets), 3, image_size, image_size))
        positives = np.zeros((len(batch_triplets), 3, image_size, image_size))
        negatives = np.zeros((len(batch_triplets), 3, image_size, image_size))

        for j, (anchor_path, pos_path, neg_path) in enumerate(batch_triplets):
            anchors[j] = preprocess_image(anchor_path, image_size)
            positives[j] = preprocess_image(pos_path, image_size)
            negatives[j] = preprocess_image(neg_path, image_size)

        x_data = {
            'anchor': anchors,
            'anchorPositive': positives,
            'anchorNegative': negatives
        }

        y_dummy = np.zeros((len(batch_triplets), 3, image_size))
        yield x_data, y_dummy


def split_dataset_folder_structure(
    source_dir, output_dir,
    train_ratio=0.7, val_ratio=0.3
):
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Tổng tỉ lệ phải bằng 1"

    class_names = os.listdir(source_dir)
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < 2:
            print(f"Bỏ qua class '{class_name}' vì quá ít ảnh.")
            continue

        # Shuffle và chia
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)

        for split_name, split_list in zip(['train', 'val'], [train_imgs, val_imgs]):
            target_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for fname in split_list:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(target_dir, fname)
                shutil.copy(src, dst)

    print("✅ Dataset đã được chia thành train/val!")

def select_semi_hard_triplets(embeddings, labels, alpha=0.2):
    """
    Chọn semi-hard triplets từ batch embeddings.
    - embeddings: [N, embedding_dim]
    - labels: [N]
    Trả về: list các tuple (anchor_idx, positive_idx, negative_idx)
    """
    triplets = []
    num_samples = len(labels)

    # Tính khoảng cách Euclidean giữa các embeddings
    dists = np.sum(np.square(embeddings[:, np.newaxis] - embeddings), axis=-1)

    for i in range(num_samples):
        anchor_label = labels[i]
        anchor_emb = embeddings[i]

        # Chọn positive candidates
        pos_indices = np.where(labels == anchor_label)[0]
        pos_indices = pos_indices[pos_indices != i]  # Loại trừ chính nó

        neg_indices = np.where(labels != anchor_label)[0]

        for pos_idx in pos_indices:
            d_ap = dists[i, pos_idx]
            for neg_idx in neg_indices:
                d_an = dists[i, neg_idx]
                if d_ap < d_an < d_ap + alpha:
                    triplets.append((i, pos_idx, neg_idx))
                    break  # Chọn 1 negative phù hợp là đủ

    return triplets


def online_triplet_generator(model, image_paths, labels, graph,type,triplets_count, batch_size= BATCH_SIZE, image_size=96, alpha=0.2):
    """
    Generator tạo batch từ dữ liệu và semi-hard triplet online mining.
    """

    label_to_paths = {}
    for path, label in zip(image_paths, labels):
        label_to_paths.setdefault(label, []).append(path)

    all_labels = list(label_to_paths.keys())

    while True:
        # 1. Chọn ngẫu nhiên batch_size ảnh
        selected_paths = []
        selected_labels = []
        for _ in range(batch_size):
            label = random.choice(all_labels)
            path = random.choice(label_to_paths[label])
            selected_paths.append(path)
            selected_labels.append(label)

        # 2. Tiền xử lý và tính embedding
        images = np.zeros((batch_size, 3, image_size, image_size))
        for i, path in enumerate(selected_paths):
            images[i] = preprocess_image(path, image_size)
        with graph.as_default():
            embeddings = model.predict(images)

        # 3. Chọn semi-hard triplets
        triplet_indices = select_semi_hard_triplets(embeddings, np.array(selected_labels), alpha)
        triplets_count.append(len(triplet_indices))

        if len(triplet_indices) == 0:
            continue  # Nếu không có triplet hợp lệ, skip batch

        # 4. Tạo batch triplet image
        anchors = np.zeros((len(triplet_indices), 3, image_size, image_size))
        positives = np.zeros_like(anchors)
        negatives = np.zeros_like(anchors)

        for i, (a, p, n) in enumerate(triplet_indices):
            anchors[i] = images[a]
            positives[i] = images[p]
            negatives[i] = images[n]

        x_data = {
            'anchor': anchors,
            'anchorPositive': positives,
            'anchorNegative': negatives
        }

        y_dummy = np.zeros((len(triplet_indices), 3, image_size))
        yield x_data, y_dummy

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img


def augment_and_save(source_dir, aug_dir, augmentations_per_image=3, image_size=96):
    datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        brightness_range=[0.5, 1.4],
        zoom_range=[0.7, 1.0]
    )

    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir, exist_ok=True)

    # Lấy danh sách tất cả ảnh cần xử lý (cho tqdm tổng thể)
    all_images = []
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path): continue
        for fname in os.listdir(class_path):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append((class_name, fname))

    # Duyệt từng ảnh với progress bar
    for class_name, fname in tqdm(all_images, desc="Đang augment ảnh"):
        class_path = os.path.join(source_dir, class_name)
        img_path = os.path.join(class_path, fname)

        target_class_path = os.path.join(aug_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        img = load_img(img_path, target_size=(image_size, image_size))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Save ảnh gốc
        original_target = os.path.join(target_class_path, f'original_{fname}')
        img.save(original_target)

        # Sinh ảnh augment
        for i, batch in enumerate(datagen.flow(x, batch_size=1)):
            aug_img = array_to_img(batch[0], scale=True)
            aug_img.save(os.path.join(target_class_path, f'aug_{i}_{fname}'))
            if i + 1 >= augmentations_per_image:
                break

    print("Augmentation hoàn tất!")


def split_dataset_folder_structure(
    source_dir, output_dir,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tổng tỉ lệ phải bằng 1"

    class_names = os.listdir(source_dir)
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < 2:
            print(f"Bỏ qua class '{class_name}' vì quá ít ảnh.")
            continue

        # Shuffle và chia
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        for split_name, split_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            target_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for fname in split_list:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(target_dir, fname)
                shutil.copy(src, dst)

    print("✅ Dataset đã được chia thành train/val/test!")

# 🧪 Gọi hàm

def count_images_in_folder(dir):
    total_images = 0
    class_counts = {}

    for class_name in os.listdir(dir):
        class_path = os.path.join(dir, class_name)
        if os.path.isdir(class_path):
            image_count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            class_counts[class_name] = image_count
            total_images += image_count


    print(f"Tổng số ảnh trong set: {total_images}")
