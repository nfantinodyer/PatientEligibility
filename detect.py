import os
import time
import argparse
import random

import pyautogui
import cv2
import numpy as np
from pywinauto import Desktop
from ultralytics import YOLO

##################################################
# CONFIGURATION
##################################################
DATA_YAML = "data.yaml"                # YOLO dataset config (train/val)
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"      # your YOLO model

# Where newly labeled images go
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"

##################################################
# TRAIN (fresh or short pass)
##################################################
def short_train(model_path, epochs=25):
    """
    We do a short training pass after adding new labeled data.
    Uses your data.yaml with train/val.
    """
    model = YOLO(model_path)  # load existing best.pt or last checkpoint
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=1920,
        batch=10,
        project="runs/detect",
        name="train",
        exist_ok=True,
        resume=False  # fresh short pass
    )
    print("[INFO] Short training pass done. Check runs/detect/train/weights/ for updated weights.")

##################################################
# CAPTURE WINDOW OR SCREEN
##################################################
def capture_window_or_screen(window_title="PlateUp!"):
    """
    Attempt to capture a window by partial title with pywinauto Desktop.
    If not found, fallback to full-screen screenshot with pyautogui.
    Returns an OpenCV BGR image (numpy).
    """
    try:
        appwin = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        pil_img = appwin.capture_as_image()
        if pil_img is None:
            raise ValueError("capture_as_image() returned None.")
        print(f"[INFO] Captured window with title containing '{window_title}'.")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[WARN] Could not capture '{window_title}' window: {e}")
        print("[WARN] Falling back to full-screen screenshot with pyautogui.")
        sc = pyautogui.screenshot()
        return cv2.cvtColor(np.array(sc), cv2.COLOR_RGB2BGR)

##################################################
# MAIN DETECTION LOGIC
##################################################
def detect_and_label(target_class="PlateUp!"):
    """
    1) Load YOLO model
    2) Capture screenshot (window or full screen)
    3) Inference:
         - If found => show bounding box, move mouse
         - If not found => user draws box => save => short train
         - If user corrects => short train
    """

    # Make sure we have a YOLO model
    best_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    if not os.path.exists(best_path):
        print(f"[ERROR] No model found at {best_path}. Please train a model first.")
        return
    print(f"[INFO] Loading YOLO model => {best_path}")
    model = YOLO(best_path)

    # Capture screenshot
    screenshot_cv = capture_window_or_screen(window_title=target_class)

    # Inference on the screenshot
    results_list = model.predict(source=screenshot_cv, imgsz=1920)
    if not results_list:
        # if YOLO returns nothing, ask user to label
        print("[WARN] No YOLO results returned. Manually label in appropriate folder.")
        user_label_val(screenshot_cv, target_class_id=0)
        short_train(best_path, epochs=25)
        return

    results = results_list[0]
    found_box = None

    # Debug: Print out detection details
    if hasattr(results, 'boxes'):
        print(f"[DEBUG] Detected {len(results.boxes)} box(es).")
        for box in results.boxes:
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            print(f"[DEBUG] Box: {box.xyxy}, Confidence: {conf:.2f}, Class ID: {cls_id}")
    else:
        print("[WARN] Model results do not contain 'boxes'. Possibly older YOLO version?")

    # Let's display all bounding boxes for clarity
    display_img = screenshot_cv.copy()
    if hasattr(results, 'boxes'):
        for box in results.boxes:
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # draw rectangle
            color = (0, 255, 0) if cls_name == target_class else (0, 0, 255)
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label_str = f"{cls_name} {conf:.2f}"
            cv2.putText(display_img, label_str, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print("[WARN] Model results do not contain 'boxes'. Possibly older YOLO version?")

    # Show detection preview for 5 seconds
    cv2.imshow("Detection Preview", display_img)
    cv2.waitKey(5000)
    cv2.destroyWindow("Detection Preview")

    # Search specifically for the target_class in the detection results
    if hasattr(results, 'boxes'):
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names[cls_id]
            if cls_name == target_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                found_box = (x1, y1, x2, y2)
                break

    # If not found => manual labeling => short train
    if not found_box:
        print(f"[INFO] No detection for '{target_class}'. Let's label it manually in appropriate folder.")
        user_label_val(screenshot_cv, target_class_id=0)
        print("[INFO] Doing short training pass with newly labeled image.")
        short_train(best_path, epochs=25)
        return

    # If found => move mouse to bounding box center
    x1, y1, x2, y2 = found_box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    print(f"[INFO] Detected '{target_class}' => center=({cx}, {cy})")

    pyautogui.moveTo(cx, cy, duration=1)
    print(f"[INFO] Moved mouse => ({cx}, {cy})")

    # Ask user if correct
    ans = input("Is mouse correct? (y/n): ").lower()
    if ans == 'y':
        print(f"[INFO] Confirmed correct => would click on '{target_class}' now.")
        # optionally do: pyautogui.click()
    else:
        # User manually corrects => short train
        print("[INFO] Please draw the correct bounding box. Press 's' to save.")
        user_label_val(screenshot_cv, target_class_id=0)
        print("[INFO] Doing short training pass with newly corrected label.")
        short_train(best_path, epochs=25)

##################################################
# USER LABEL (Train/Val Folder)
##################################################
drawing = False
sx, sy, ex, ey = -1, -1, -1, -1

def mouse_callback(event, x, y, flags, param):
    global drawing, sx, sy, ex, ey
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y
        ex, ey = x, y
        print(f"[DEBUG] Mouse down at ({sx}, {sy})")
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y
        print(f"[DEBUG] Mouse move to ({ex}, {ey})")
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        print(f"[DEBUG] Mouse up at ({ex}, {ey})")

def user_label_val(image_cv, target_class_id=0):
    """
    Opens an OpenCV window => user draws bounding box => press 's' to save,
    'r' to reset, or 'q' to skip. Then saves to either train (80%) or val (20%) folder.
    """
    # Randomly decide folder: 80% for train, 20% for val
    folder = "train" if random.random() < 0.8 else "val"
    if folder == "train":
        img_dir = TRAIN_IMG_DIR
        label_dir = TRAIN_LABEL_DIR
    else:
        img_dir = VAL_IMG_DIR
        label_dir = VAL_LABEL_DIR

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    wname = f"Label {folder.upper()} - draw box, press 's' to save, 'r' to reset, 'q' to skip"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, mouse_callback)

    global sx, sy, ex, ey
    sx = sy = ex = ey = -1

    clone = image_cv.copy()
    while True:
        temp = clone.copy()
        if sx != -1 and sy != -1 and ex != -1 and ey != -1:
            cv2.rectangle(temp, (sx, sy), (ex, ey), (0, 255, 0), 2)
            print(f"[DEBUG] Drawing rectangle: ({sx}, {sy}) to ({ex}, {ey})")

        cv2.imshow(wname, temp)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('r'):
            # reset
            print("[INFO] Reset bounding box.")
            sx = sy = ex = ey = -1

        elif key == ord('s'):
            if sx == ex or sy == ey:
                print("[WARN] No bounding box drawn => try again.")
                continue
            # Save image
            tstamp = int(time.time())
            img_name = f"{folder}_user_{tstamp}.jpg"
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, image_cv)
            print(f"[INFO] Saved {folder.upper()} image => {img_path}")

            # YOLO label
            h, w = image_cv.shape[:2]
            xx1, xx2 = sorted([sx, ex])
            yy1, yy2 = sorted([sy, ey])
            bw = xx2 - xx1
            bh = yy2 - yy1
            x_center = (xx1 + bw/2) / w
            y_center = (yy1 + bh/2) / h
            norm_w = bw / w
            norm_h = bh / h

            label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as lf:
                line = f"{target_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line + "\n")

            print(f"[INFO] Saved {folder.upper()} label => {label_path}")
            print(f"[DEBUG] Label contents => {line}")
            cv2.destroyWindow(wname)
            return True

        elif key == ord('q'):
            print("[INFO] Skipped labeling => no new data saved.")
            cv2.destroyWindow(wname)
            return False

##################################################
# MAIN
##################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="PlateUp!",
                        help="Which class to detect/correct.")
    args = parser.parse_args()

    while True:
        detect_and_label(args.target)

if __name__ == "__main__":
    main()
