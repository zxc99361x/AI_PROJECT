import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D ,Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 資料前處理
x_train = x_train / 255.0  # 正規化
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # One-hot 編碼
y_test = to_categorical(y_test, 10)

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 展平圖像
    Dense(256, activation='relu'),  # 第一層隱藏層，256 個神經元
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),  # 第二層隱藏層，128 個神經元
    Dense(64, activation='relu'),   # 第三層隱藏層，64 個神經元
    Dropout(0.2),
    Dense(10, activation='softmax')  # 輸出層
])

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001),  # 調整學習率
              loss='categorical_crossentropy',
              metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=15,  # 隨機旋轉
    width_shift_range=0.2,  # 水平移動
    height_shift_range=0.2,  # 垂直移動
    zoom_range=0.2  # 隨機縮放
)

# 訓練模型
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=100, validation_data=(x_test, y_test), validation_split=0.2)

# 保存模型
model.save("mnist_model.h5")
print("模型保存完成！")

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print(f"測試集準確率: {test_acc * 100:.2f}%")