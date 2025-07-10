import cv2
import os

def capture_images(person_name, num_images=5, base_dir="images"):
    # Tạo thư mục lưu ảnh
    save_dir = os.path.join(base_dir, person_name)
    os.makedirs(save_dir, exist_ok=True)

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    print(f"Nhấn phím 'c' để chụp ảnh ({num_images} ảnh), 'q' để thoát.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame.")
            break

        # Hiển thị khung hình
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Lưu ảnh
            img_name = f"img_{count+1}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"[{count+1}/{num_images}] Đã lưu ảnh: {img_path}")
            count += 1

        elif key == ord('q'):
            print("Đã thoát trước khi chụp đủ ảnh.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất.")

if __name__ == "__main__":
    name = input("Nhập tên người (tên folder): ").strip()
    if name:
        capture_images(name)
    else:
        print("Tên không hợp lệ.")
