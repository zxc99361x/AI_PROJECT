import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import random

# 載入 EMNIST Balanced 數據集
(train_ds, test_ds), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# 過濾數據：僅保留字母類別
def filter_letters(dataset):
    def is_letter(image, label):
        return tf.logical_and(label >= 10, label < 36)
    return dataset.filter(is_letter)

train_ds = filter_letters(train_ds)
test_ds = filter_letters(test_ds)

# 預處理數據
def preprocess(image, label, augment=False):
    image = tf.image.resize(image, (28, 28)) / 255.0
    if augment:  # 僅在訓練集啟用增強
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.expand_dims(image, axis=-1)  # 添加通道維度
    label = label - 10  # 調整標籤
    return image, label

# 對訓練集啟用增強，測試集僅做基本預處理
train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True)).batch(32)
test_ds = test_ds.map(lambda x, y: preprocess(x, y, augment=False)).batch(32)

# 字母映射表
emnist_mapping = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# 建立模型
model = Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 學習率調度
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 訓練模型
model.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[reduce_lr])

# 保存模型
model.save("emnist_letters_model.h5")

# 測試集準確率
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"測試集損失：{test_loss:.4f}")
print(f"測試集準確率：{test_accuracy:.4f}")

# 隨機顯示測試數據的預測結果
def visualize_random_predictions(dataset, model, mapping, num_samples=10):
    # 將數據集展開為列表以支持隨機抽樣
    data_list = list(dataset.unbatch().as_numpy_iterator())
    sampled_data = random.sample(data_list, num_samples)  # 隨機抽取樣本

    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(sampled_data):
        # 預測
        prediction = model.predict(tf.expand_dims(image, axis=0))
        predicted_label = np.argmax(prediction)
        
        # 繪製圖像和標籤
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"真實: {mapping[label]}\n預測: {mapping[predicted_label]}")
        plt.axis('off')
    plt.show()

# 隨機抽取並可視化測試集的預測結果
visualize_random_predictions(test_ds, model, emnist_mapping, num_samples=20)
