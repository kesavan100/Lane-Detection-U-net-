import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image

# Load models once
lane_model = load_model("lane_unet_model.h5")
yolo_model = YOLO("yolov8n.pt")
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
names = yolo_model.model.names

# ---------------------
# Preprocessing functions
# ---------------------
def preprocess_uploaded_image(uploaded_file, img_size=(128, 128)):
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_bgr, img_size) / 255.0
    return img_bgr, np.expand_dims(resized, axis=0)

def postprocess_lane_mask(mask):
    mask = (mask > 127).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # remove noise
    
    # Optional thinning if ximgproc available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        mask = cv2.ximgproc.thinning(mask)
    return mask

# ---------------------
# Lane detection functions
# ---------------------
def get_lane_line_points(mask, min_lane_width=30, step=5, roi_height_ratio=0.4):
    h, w = mask.shape
    center_x = w // 2
    start_y = int(h * (1 - roi_height_ratio))
    left_points, right_points = [], []

    for y in range(h - 1, start_y, -step):
        row = mask[y]
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) < 2:
            continue

        left = white_pixels[white_pixels < center_x]
        right = white_pixels[white_pixels > center_x]

        if len(left) > 0:
            left_points.append((np.max(left), y))
        if len(right) > 0:
            right_points.append((np.min(right), y))

    if len(left_points) > 5:
        left_points = reject_outliers(left_points)
    if len(right_points) > 5:
        right_points = reject_outliers(right_points)
    return left_points, right_points

def reject_outliers(points, threshold=2.0):
    if len(points) < 3:
        return points
    x_coords = np.array([p[0] for p in points])
    median = np.median(x_coords)
    mad = 1.4826 * np.median(np.abs(x_coords - median))
    return [p for p in points if abs(p[0] - median) < threshold * mad]

# ---------------------
# Lane visualization (Updated)
# ---------------------
def highlight_current_lane_area(image, left_points, right_points):
    overlay = image.copy()
    mask = np.zeros_like(image)

    if len(left_points) < 5 or len(right_points) < 5:
        return image

    h, w, _ = image.shape

    # Sort by y-coordinate for consistent polygon
    left_points_sorted = sorted(left_points, key=lambda p: p[1])
    right_points_sorted = sorted(right_points, key=lambda p: p[1])

    # Ensure same y range to avoid polygon bleeding into adjacent lanes
    min_y = max(min(p[1] for p in left_points_sorted), min(p[1] for p in right_points_sorted))
    max_y = min(max(p[1] for p in left_points_sorted), max(p[1] for p in right_points_sorted))

    left_clamped = [(x, y) for x, y in left_points_sorted if min_y <= y <= max_y]
    right_clamped = [(x, y) for x, y in right_points_sorted if min_y <= y <= max_y]

    if len(left_clamped) < 2 or len(right_clamped) < 2:
        return image

    # Create polygon strictly between left and right lane lines
    pts = np.array(left_clamped + right_clamped[::-1], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (0, 200, 0))
    return cv2.addWeighted(overlay, 0.8, mask, 0.2, 0)

# ---------------------
# Lane region calculation
# ---------------------
def get_lane_regions(left_pts, right_pts, img_shape, safety_margin=50):
    h, w, _ = img_shape
    if not left_pts or not right_pts:
        return None, None, None
    left_x = [p[0] for p in left_pts]
    left_y = [p[1] for p in left_pts]
    right_x = [p[0] for p in right_pts]
    right_y = [p[1] for p in right_pts]
    left_fit = np.polyfit(left_y, left_x, 2) if len(left_pts) > 5 else None
    right_fit = np.polyfit(right_y, right_x, 2) if len(right_pts) > 5 else None
    current_lane = {
        'left_boundary': left_pts,
        'right_boundary': right_pts,
        'left_fit': left_fit,
        'right_fit': right_fit,
        'y_top': min(left_pts[-1][1], right_pts[-1][1]),
        'y_bottom': h
    }
    left_adjacent = None
    if left_fit is not None:
        left_adjacent = {
            'left_boundary': [(max(0, x - safety_margin), y) for x, y in left_pts],
            'right_boundary': left_pts,
            'left_fit': left_fit,
            'right_fit': None,
            'y_top': current_lane['y_top'],
            'y_bottom': h
        }
    right_adjacent = None
    if right_fit is not None:
        right_adjacent = {
            'left_boundary': right_pts,
            'right_boundary': [(min(w, x + safety_margin), y) for x, y in right_pts],
            'left_fit': None,
            'right_fit': right_fit,
            'y_top': current_lane['y_top'],
            'y_bottom': h
        }
    return current_lane, left_adjacent, right_adjacent

# ---------------------
# Vehicle detection
# ---------------------
def detect_vehicles(image, yolo_model):
    results = yolo_model(image)[0]
    vehicles = []
    for box in results.boxes:
        try:
            cls_id = int(box.cls[0])
        except:
            cls_id = int(box.cls)
        label = names[cls_id]
        if label in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'confidence': confidence,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'bottom_center': ((x1 + x2) // 2, y2)
            })
    return vehicles

# ---------------------
# Lane change decision
# ---------------------
# ---------------------
# Lane change decision (Fixed: Checks if lane actually exists)
# ---------------------
def make_lane_change_decision(current_lane, left_adjacent, right_adjacent, vehicles, image_shape):
    h, w = image_shape[:2]
    decision = {'text': "No lane change advice available", 'color': (0, 0, 255),
                'left_clear': False, 'right_clear': False, 'reason': "Insufficient lane data"}

    if not current_lane or not vehicles:
        return decision

    MIN_CLEARANCE = 250
    LANE_WIDTH_MULTIPLIER = 1.5

    if current_lane['left_fit'] is not None and current_lane['right_fit'] is not None:
        lane_width = abs(np.polyval(current_lane['right_fit'], h) - np.polyval(current_lane['left_fit'], h))
        safety_width = lane_width * LANE_WIDTH_MULTIPLIER
    else:
        lane_width = w // 3
        safety_width = lane_width

    # ---------- LEFT LANE CHECK ----------
    if left_adjacent is None or left_adjacent['left_fit'] is None:
        left_clear = False
        left_reason = "No left lane detected"
    else:
        left_clear = True
        left_reason = "Clear"
        for vehicle in vehicles:
            vx, vy = vehicle['bottom_center']
            if vy < h // 2:
                continue
            left_bound = np.polyval(left_adjacent['left_fit'], vy)
            right_bound = np.polyval(current_lane['left_fit'], vy)
            if left_bound - safety_width < vx < right_bound and vy > h - MIN_CLEARANCE:
                left_clear = False
                left_reason = f"{vehicle['label']} detected in left lane"
                break

    if right_adjacent is None or right_adjacent['right_fit'] is None:
        right_clear = False
        right_reason = "No right lane detected"
    else:
        right_clear = True
        right_reason = "Clear"
        for vehicle in vehicles:
            vx, vy = vehicle['bottom_center']
            if vy < h // 2:
                continue
            left_bound = np.polyval(current_lane['right_fit'], vy)
            right_bound = np.polyval(right_adjacent['right_fit'], vy)
            if left_bound < vx < right_bound + safety_width and vy > h - MIN_CLEARANCE:
                right_clear = False
                right_reason = f"{vehicle['label']} detected in right lane"
                break

    if left_clear and right_clear:
        decision = {'text': "✅ Both lanes clear to overtake(choose left)", 'color': (0, 255, 0),
                    'left_clear': True, 'right_clear': True, 'reason': "No vehicles detected in adjacent lanes"}
    elif left_clear:
        decision = {'text': "⬅ Left lane clear to overtake", 'color': (0, 255, 0),
                    'left_clear': True, 'right_clear': False, 'reason': left_reason}
    elif right_clear:
        decision = {'text': "➡ Right lane clear to overtake", 'color': (0, 255, 0),
                    'left_clear': False, 'right_clear': True, 'reason': right_reason}
    else:
        decision = {'text': "🛑 No safe lane change possible to Overtake", 'color': (0, 0, 255),
                    'left_clear': False, 'right_clear': False, 'reason': f"Left: {left_reason} | Right: {right_reason}"}

    return decision


# ---------------------
# Visualization
# ---------------------
def draw_detailed_visualization(image, current_lane, left_adjacent, right_adjacent, vehicles, decision):
    debug_img = image.copy()
    if current_lane and len(current_lane['left_boundary']) > 2 and len(current_lane['right_boundary']) > 2:
        pts = np.array(current_lane['left_boundary'] + current_lane['right_boundary'][::-1], dtype=np.int32)
        cv2.polylines(debug_img, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    if left_adjacent and len(left_adjacent['left_boundary']) > 2:
        pts = np.array(left_adjacent['left_boundary'] + left_adjacent['right_boundary'][::-1], dtype=np.int32)
        cv2.polylines(debug_img, [pts.reshape((-1, 1, 2))], True, (255, 0, 0), 1)
    if right_adjacent and len(right_adjacent['right_boundary']) > 2:
        pts = np.array(right_adjacent['left_boundary'] + right_adjacent['right_boundary'][::-1], dtype=np.int32)
        cv2.polylines(debug_img, [pts.reshape((-1, 1, 2))], True, (0, 0, 255), 1)
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        color = (0, 165, 255)
        if vehicle['bottom_center'][1] > image.shape[0] - 250:
            color = (0, 0, 255)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_img, f"{vehicle['label']} {vehicle['confidence']:.1f}", 
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(decision['text'], font, font_scale, thickness)
    cv2.rectangle(debug_img, (10, 10), (20 + text_width, 20 + text_height), (0, 0, 0), -1)
    cv2.putText(debug_img, decision['text'], (20, 20 + text_height), font, font_scale, decision['color'], thickness, cv2.LINE_AA)
    return debug_img

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(layout="wide", page_title="Advanced Lane Detection", page_icon="🚦")
st.title("🚦 Advanced Lane Detection and Lane Change Advisor")
st.markdown("Upload a road image to detect lanes and get accurate lane change advice")

uploaded_file = st.file_uploader("📂 Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.success("✅ Image uploaded successfully! Processing...")
    orig_img, processed_img = preprocess_uploaded_image(uploaded_file)
    lane_pred = lane_model.predict(processed_img, verbose=0)[0]
    lane_mask = (lane_pred > 0.5).astype(np.uint8) * 255
    lane_mask = cv2.resize(lane_mask, (orig_img.shape[1], orig_img.shape[0]))
    lane_mask = postprocess_lane_mask(lane_mask)  # New post-processing
    left_pts, right_pts = get_lane_line_points(lane_mask)
    highlighted_img = highlight_current_lane_area(orig_img.copy(), left_pts, right_pts)
    vehicles = detect_vehicles(orig_img, yolo_model)
    current_lane, left_adjacent, right_adjacent = get_lane_regions(left_pts, right_pts, orig_img.shape)
    decision = make_lane_change_decision(current_lane, left_adjacent, right_adjacent, vehicles, orig_img.shape)
    debug_img = draw_detailed_visualization(highlighted_img, current_lane, left_adjacent, right_adjacent, vehicles, decision)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                 caption="📷 Advanced Lane Analysis", use_column_width=True)
    with col2:
        st.image(lane_mask, caption="🧠 Processed Lane Mask", use_column_width=True)
    st.info(f"🚘 **Decision:** `{decision['text']}`")
    st.markdown(f"**Reason:** {decision['reason']}")
    with st.expander("📊 Technical Details"):
        st.write(f"- Detected left lane points: {len(left_pts)}")
        st.write(f"- Detected right lane points: {len(right_pts)}")
        st.write(f"- Total vehicles detected: {len(vehicles)}")
        close_vehicles = [v for v in vehicles if v['bottom_center'][1] > orig_img.shape[0] - 250]
        st.write(f"- Vehicles in close proximity: {len(close_vehicles)}")
        st.write(f"- Left lane status: {'Clear' if decision['left_clear'] else 'Occupied'}")
        st.write(f"- Right lane status: {'Clear' if decision['right_clear'] else 'Occupied'}")
else:
    st.warning("⚠️ Please upload a valid road image to start.")
