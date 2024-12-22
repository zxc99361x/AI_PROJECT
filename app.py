from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# 載入模型
model = tf.keras.models.load_model("mnist_model.h5")
emnist_model = tf.keras.models.load_model("emnist_model.h5")


emnist_mapping = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def preprocess_image(image):
    # 確保縮放到 28x28 並轉為灰階
    image = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    
    # 可選：檢查是否需要反轉顏色（畫布可能是黑底白字）
    image = np.invert(image)

    # 轉為數組並進行歸一化
    image = np.array(image).astype('float32') / 255.0  # 確保像素值在 [0, 1]
    image = np.expand_dims(image, axis=(0, -1))  # 添加模型需要的批次和通道維度

    # 保存預處理後的圖像
    plt.imsave("preprocessed_image.png", image.reshape(28, 28), cmap='gray')
    print("處理後的圖像矩陣：")
    print(image.reshape(28, 28))

    return image


@app.route('/')
def index():
    return render_template('index.html')    # 原 MNIST 頁面

@app.route('/emnist')
def emnist():
    return render_template('emnist.html')  # 新 EMNIST 頁面

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 接收和預處理圖像
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image.save("received_image.png")  # 保存原始接收圖像

        processed_image = preprocess_image(image)  # 預處理圖像

        # 打印處理後的圖像數據
        print("傳入模型的數據：", processed_image)

        # 使用模型進行預測
        prediction = model.predict(processed_image)
        print("預測機率分布：", prediction)

        result = int(np.argmax(prediction))
        return jsonify({'result': result})
    except Exception as e:
        print(f"錯誤：{e}")
        return jsonify({'error': str(e)})

@app.route('/predict_emnist', methods=['POST'])
def predict_emnist():
    try:
        # 接收前端數據
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)

        # 模型預測
        prediction = emnist_model.predict(processed_image)
        class_index = int(np.argmax(prediction))  # 獲取索引

        # 映射索引到字母
        if 0 <= class_index < len(emnist_mapping):
            predicted_character = emnist_mapping[class_index]
        else:
            predicted_character = '?'

        return jsonify({'result': predicted_character})
    except Exception as e:
        print(f"錯誤：{e}")
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
