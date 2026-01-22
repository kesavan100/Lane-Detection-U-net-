import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# 1. Load Dataset
# ===============================
IMG_SIZE = (128, 128)
img_dir = r"D:\D_downloads\lane_data_img\frames"
mask_dir = r"D:\D_downloads\lane_data_img\lane-masks"

images, masks = [], []
for file in sorted(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, file)
    mask_path = os.path.join(mask_dir, file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    img = cv2.resize(img, IMG_SIZE) / 255.0
    mask = cv2.resize(mask, IMG_SIZE) / 255.0
    mask = np.expand_dims(mask, axis=-1)

    images.append(img.astype(np.float32))
    masks.append(mask.astype(np.float32))

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

def build_unet(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)
    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs, name="U-Net")

def build_scnn(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="SCNN")

def build_mobilenet_seg(input_shape=(128, 128, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    skip1 = base_model.get_layer("block_1_expand_relu").output
    skip2 = base_model.get_layer("block_3_expand_relu").output
    skip3 = base_model.get_layer("block_6_expand_relu").output
    encoder_output = base_model.output

    x = UpSampling2D((2, 2))(encoder_output)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip3])
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip2])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip1])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=outputs, name="MobileNet_Seg")

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0

histories = {}
models = {}
metrics_results = {"Accuracy": {}, "IoU": {}, "Dice": {}}

for name, builder in [("U-Net", build_unet), ("SCNN", build_scnn), ("MobileNet", build_mobilenet_seg)]:
    print(f"\nTraining {name}...")
    model = builder((128, 128, 3))
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=8,
        verbose=1
    )
    models[name] = model
    histories[name] = history
    model.save(f"{name.lower().replace('-', '')}_model.h5")

    preds = model.predict(X_test, verbose=0)
    preds_binary = (preds > 0.5).astype(np.uint8)
    y_true_binary = (y_test > 0.5).astype(np.uint8)

    acc = accuracy_score(y_true_binary.flatten(), preds_binary.flatten())
    iou = iou_score(y_true_binary, preds_binary)
    dice = dice_score(y_true_binary, preds_binary)

    metrics_results["Accuracy"][name] = acc
    metrics_results["IoU"][name] = iou
    metrics_results["Dice"][name] = dice

plt.figure(figsize=(8, 5))
for name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{name} Train')
    plt.plot(history.history['val_accuracy'], label=f'{name} Val')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

metrics_names = list(metrics_results.keys())
x = np.arange(len(models))
bar_width = 0.25

plt.figure(figsize=(8, 5))
for i, metric in enumerate(metrics_names):
    values = [metrics_results[metric][name] for name in models.keys()]
    plt.bar(x + i * bar_width, values, width=bar_width, label=metric)

plt.xticks(x + bar_width, models.keys())
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Comparison: Accuracy, IoU, Dice")
plt.legend()
plt.grid(axis='y')
plt.show()
