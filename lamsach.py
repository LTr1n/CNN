import os
import cv2
import hashlib
import numpy as np
from tqdm import tqdm

# Thư mục dữ liệu
train_dir = "train"  # Cập nhật đường dẫn đúng
val_dir = "val"
test_dir = "test"
image_size = (128, 128)  # Chuẩn hóa kích thước ảnh
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}  # Định dạng ảnh hợp lệ

def get_image_hash(image):
    """Tạo hash từ dữ liệu ảnh để kiểm tra trùng lặp."""
    return hashlib.md5(image.tobytes()).hexdigest()

def clean_images(directory):
    """Làm sạch ảnh trong thư mục chỉ định, bao gồm kiểm tra trùng lặp."""
    hashes = set()
    
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc=f"Processing {directory}"):
            img_path = os.path.join(root, file)
            
            # Bỏ qua file không phải ảnh
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                print(f"❌ Bỏ qua file không phải ảnh: {img_path}")
                continue
            
            try:
                # Đọc ảnh bằng OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"❌ Ảnh hỏng, xóa: {img_path}")
                    os.remove(img_path)
                    continue
                
                # Kiểm tra số kênh ảnh (chuyển về RGB nếu có kênh alpha)
                if len(img.shape) == 2:  # Ảnh grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # Ảnh RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                # Kiểm tra kích thước ảnh
                height, width, _ = img.shape
                if height < 32 or width < 32:  # Loại bỏ ảnh quá nhỏ
                    print(f"⚠️ Ảnh quá nhỏ, xóa: {img_path}")
                    os.remove(img_path)
                    continue
                
                # Kiểm tra ảnh trùng lặp
                img_hash = get_image_hash(img)
                if img_hash in hashes:
                    print(f"🔄 Ảnh trùng lặp, xóa: {img_path}")
                    os.remove(img_path)
                    continue
                hashes.add(img_hash)
                
                # Chuẩn hóa kích thước bằng OpenCV
                img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_path, img_resized)
            
            except Exception as e:
                print(f"❌ Lỗi với ảnh {img_path}: {e}")
                os.remove(img_path)

# Chạy làm sạch cho cả train & val
clean_images(train_dir)
clean_images(val_dir)
clean_images(test_dir)
print("✅ Xong!")
