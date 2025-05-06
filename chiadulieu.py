import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục Animals-10
data_dir = "Animals-10"  

# Đường dẫn đến thư mục đích để lưu train, val, test
base_dir = "/Users/macbook/Desktop/Hoc tap/Dữ liệu lớn và học sâu" 
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Tạo các thư mục train, val, test nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Danh sách các lớp (dựa trên hình ảnh bạn cung cấp)
classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Duyệt qua từng lớp để chia dữ liệu
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        print(f"Thư mục {class_name} không tồn tại, bỏ qua...")
        continue

    # Lấy danh sách tất cả các file ảnh trong thư mục lớp
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
              if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"Không tìm thấy ảnh trong thư mục {class_name}, bỏ qua...")
        continue

    # Chia dữ liệu: 80% train, 20% tạm thời (sẽ chia tiếp thành val và test)
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    # Chia 20% còn lại thành 10% val và 10% test
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Tạo thư mục cho từng lớp trong train, val, test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Sao chép file vào các thư mục tương ứng
    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, class_name))
    for img in val_imgs:
        shutil.copy(img, os.path.join(val_dir, class_name))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, class_name))

    # In số lượng ảnh để kiểm tra
    print(f"Lớp {class_name}:")
    print(f" - Train: {len(train_imgs)} ảnh")
    print(f" - Val: {len(val_imgs)} ảnh")
    print(f" - Test: {len(test_imgs)} ảnh")

print("Đã chia tập dữ liệu thành công!")