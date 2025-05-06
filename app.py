from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = load_model('/Users/macbook/Desktop/Hoc tap/Dữ liệu lớn và học sâu/ml/best_model.keras')
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'hourse', 'sheep', 'spider', 'squirrel']  # Cập nhật danh sách loài

UPLOAD_FOLDER = 'sz`tatic/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        predicted_class, confidence = predict_image(file_path)

        return render_template('result.html', image_path=file_path, predicted_class=predicted_class, confidence=confidence)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
