

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#from ultralytics import YOLO

lane_model = load_model("lane_unet_model.h5")
#yolo_model = YOLO("yolov8n.pt")

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

img_path = "D:\D_downloads\lane_data_img\Strada_Provinciale_BS_510_Sebina_Orientale.jpg"
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
