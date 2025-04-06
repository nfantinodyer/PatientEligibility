import os
import time
import argparse

import pyautogui
import cv2
import numpy as np
from pywinauto import Desktop
from ultralytics import YOLO

def focus_window(window_title):
    """
    Tries to find a window by regex title and bring it to the front.
    If not found, prints a warning.
    """
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        win.set_focus()
        win.maximize()
        time.sleep(0.5)  # small delay to ensure the window is in front
        print(f"[INFO] Focused window matching '{window_title}'.")
    except Exception as e:
        print(f"[WARN] Could not focus window '{window_title}': {e}")

def run_detection_for_class(
    image_cv, 
    model, 
    target_class_id, 
    conf_thres=0.5, 
    duration=2, 
    window_title="Detections"
):
    """
    Runs YOLO detection on image_cv, filtering only 'target_class_id' at the given
    confidence threshold, shows bounding boxes in a 1920×1080 window for `duration` seconds,
    and moves the mouse to the highest-confidence bounding box (if found).
    """
    # 1) Inference
    results_list = model.predict(source=image_cv, imgsz=1920)
    if not results_list:
        print("[WARN] No inference results returned.")
        return
    results = results_list[0]

    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        print(f"[INFO] No detections found for class {target_class_id}.")
        return

    # 2) Draw bounding boxes & pick highest confidence box
    display_img = image_cv.copy()
    max_conf = -1.0
    best_center = None

    for box in results.boxes:
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        # Filter out by confidence
        if conf < conf_thres:
            continue
        # Filter by class_id
        if cls_id != target_class_id:
            continue

        # Draw the box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color=(0, 255, 0), thickness=2)
        label_str = f"cls={cls_id}, conf={conf:.5f}"
        cv2.putText(display_img, label_str, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Track highest confidence box
        if conf > max_conf:
            max_conf = conf
            best_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))

    # 3) Move the mouse if we found a best box
    if best_center is not None:
        cx, cy = best_center
        print(f"[INFO] Class {target_class_id}: moving mouse => center=({cx}, {cy}), conf={max_conf:.3f}")
        pyautogui.moveTo(cx, cy, duration=1)
    else:
        print(f"[INFO] No box found above conf {conf_thres} for class {target_class_id}.")

    # 4) Show the detection results in a 1920×1080 window
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1920, 1080)
    cv2.imshow(window_title, display_img)
    focus_window(window_title)  # Ensure the window is in focus
    cv2.waitKey(duration * 3000)
    cv2.destroyWindow(window_title)
    window_title="Remote Control"
    focus_window(window_title)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/rm7/weights/best.pt",
        help="Path to YOLO model weights (default: runs/detect/rm7/weights/best.pt)"
    )
    parser.add_argument(
        "--window",
        type=str,
        default="Remote Control",
        help="Window title (partial) to focus before screenshot. Leave blank for none."
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default=0.5)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=2,
        help="How many seconds to show the detection window each time (default=2)."
    )
    args = parser.parse_args()

    # 1) Focus the desired window (if any)
    if args.window.strip():
        focus_window(args.window.strip())

    # 2) Screenshot the entire screen
    screenshot = pyautogui.screenshot()
    image_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]
    print(f"[INFO] Captured screenshot: {w}x{h}")

    # 3) Load YOLO model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights not found: {args.model}")
        return
    print(f"[INFO] Loading YOLO model => {args.model}")
    model = YOLO(args.model)

    # 4) Run detection for class 0
    print("[INFO] Detecting class 0 Insurance...")
    run_detection_for_class(
        image_cv, model, 
        target_class_id=0, 
        conf_thres=args.conf_thres, 
        duration=args.duration, 
        window_title="Detection - Class 0 Insurance"
    )
    """
    # 5) Wait 5 seconds
    print("[INFO] Waiting 5 seconds before next detection...")
    time.sleep(5)

    # 6) Run detection for class 1 (Factorio)
    print("[INFO] Detecting class 1 (Factorio)...")
    run_detection_for_class(
        image_cv, model, 
        target_class_id=1, 
        conf_thres=args.conf_thres, 
        duration=args.duration, 
        window_title="Detection - Class 1 (Factorio)"
    )
    """

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
