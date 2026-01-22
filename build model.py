import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

img_dir = "D:\\D_downloads\\lane_data_img\\frames"
mask_dir = "D:\\D_downloads\\lane_data_img\\lane-masks"

IMG_SIZE = (128, 128)
images, masks = [], []
for file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, file)
    mask_path = os.path.join(mask_dir, file)

    img = cv2.imread(img_path)                    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
    if img is None or mask is None:
        continue

    img = cv2.resize(img, IMG_SIZE) / 255.0
    mask = cv2.resize(mask, IMG_SIZE) / 255.0
    mask = np.expand_dims(mask, axis=-1)  

    images.append(img)
    masks.append(mask)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

def build_unet(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

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

    return Model(inputs, outputs)
model = build_unet((128, 128, 3))
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=8
)


pred_mask = model.predict(np.expand_dims(X_val[0], axis=0))[0]

print("Prediction mask shape:", pred_mask.shape)

model.save("lane_unet_model.h5")
print("Model saved as lane_unet_model.h5")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


lane_model = load_model("lane_unet_model.h5")


def load_and_preprocess_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found!")
    img_resized = cv2.resize(img, img_size) / 255.0
    return img, np.expand_dims(img_resized, axis=0)

def get_lane_line_points(lane_mask, min_lane_width=30, step=5, roi_height_ratio=0.3):
    h, w = lane_mask.shape
    center_x = w // 2
    start_y = int(h * (1 - roi_height_ratio))
    left_points, right_points = [], []

    for y in range(h - 1, start_y, -step):
        row = lane_mask[y]
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) < 2:
            continue

        left = white_pixels[white_pixels < center_x]
        right = white_pixels[white_pixels > center_x]

        if len(left) > 0 and len(right) > 0:
            left_x = np.max(left)
            right_x = np.min(right)

            if right_x - left_x > min_lane_width:
                left_points.append((left_x, y))
                right_points.append((right_x, y))

    return left_points, right_points

def highlight_lane_polygon(image, left_points, right_points):
    overlay = image.copy()
    mask = np.zeros_like(image)

    if len(left_points) < 5 or len(right_points) < 5:
        return image

    pts = np.array(left_points + right_points[::-1], dtype=np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask, [pts], (0, 255, 0))

    for pt1, pt2 in zip(left_points[:-1], left_points[1:]):
        cv2.line(mask, pt1, pt2, (255, 0, 255), 4)

    for pt1, pt2 in zip(right_points[:-1], right_points[1:]):
        cv2.line(mask, pt1, pt2, (0, 255, 255), 4)

    return cv2.addWeighted(overlay, 1.0, mask, 0.6, 0)

img_path = "D:\\D_downloads\\lane_data_img\\Screenshot 2025-08-06 140334.png"
orig_img, processed_img = load_and_preprocess_image(img_path)

lane_pred = lane_model.predict(processed_img, verbose=0)[0]
lane_mask = (lane_pred > 0.5).astype(np.uint8) * 255
lane_mask = cv2.resize(lane_mask, (orig_img.shape[1], orig_img.shape[0]))

left_pts, right_pts = get_lane_line_points(lane_mask, roi_height_ratio=0.288)

highlighted_img = highlight_lane_polygon(orig_img, left_pts, right_pts)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Lane Highlighted")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(lane_mask, cmap="gray")
plt.title("Binary Lane Mask")
plt.axis("off")
plt.show()