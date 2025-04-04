import os
import time
import json
import argparse

import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

##################################################
# CONFIGURATION
##################################################
DATA_YAML = "data.yaml"                # YOLO dataset config (train/val)
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"      # your YOLO model
OFFSET_FILE = "mouse_offsets.json"     # store user offset

# Where newly *corrected* or user-labeled images go
# For training images, use data/images/train => data/labels/train
# For *val* images (as you requested), we store them in data/images/val => data/labels/val
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"

##################################################
# OFFSET LOADING/SAVING
##################################################
def load_offsets():
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARN] Could not parse mouse_offsets.json. Returning empty.")
    return {}

def save_offsets(offsets):
    with open(OFFSET_FILE, "w") as f:
        json.dump(offsets, f, indent=4)

##################################################
# TRAIN (fresh or short pass)
##################################################
def short_train(model_path, epochs=1):
    """
    We do a short training pass after adding new labeled data.
    This uses your data.yaml, referencing train/val folders.
    """
    model = YOLO(model_path)  # load existing best.pt or last checkpoint
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=4,
        project="runs/detect",
        name="train",
        exist_ok=True,
        resume=False  # fresh short pass
    )
    print("[INFO] Short training pass done. Check runs/detect/train/weights/ for updated weights.")

##################################################
# MAIN DETECTION LOGIC
##################################################
def detect_and_label(target_class="PlateUp!"):
    """
    1) Load YOLO model
    2) Screenshot
    3) Inference
    4) If found => ask offset correction
    5) If not found => user draws box => saved in data/images/val => data/labels/val
    6) Then short train
    """

    offsets = load_offsets()
    offset_x = offsets.get(target_class, {}).get("x", 0)
    offset_y = offsets.get(target_class, {}).get("y", 0)

    # 1) Make sure we have a YOLO model
    best_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    if not os.path.exists(best_path):
        print(f"[ERROR] No model found at {best_path}. Please train a model first.")
        return
    print(f"[INFO] Loading YOLO model => {best_path}")
    model = YOLO(best_path)

    # 2) Screenshot
    screenshot = pyautogui.screenshot()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 3) Inference
    results = model(screenshot_cv)[0]
    found_box = None
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = results.names[cls_id]
        if cls_name == target_class:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            found_box = (x1, y1, x2, y2)
            break

    # 4) If detection fails => user draws box => save to val
    if not found_box:
        print(f"[INFO] No detection for '{target_class}'. Let's label it manually in val folder.")
        user_label_val(screenshot_cv, target_class_id=0)  # assume class_id=0 if single class
        print("[INFO] Doing short training pass with newly labeled val image.")
        short_train(best_path, epochs=1)
        return

    # If found => apply offset
    x1, y1, x2, y2 = found_box
    cx = int((x1 + x2)/2)
    cy = int((y1 + y2)/2)
    print(f"[INFO] Detected '{target_class}' => center=({cx}, {cy})")

    final_x = cx + offset_x
    final_y = cy + offset_y
    pyautogui.moveTo(final_x, final_y, duration=1)
    print(f"[INFO] Moved mouse => ({final_x}, {final_y})")

    # ask user if correct
    ans = input("Is mouse correct? (y/n): ").lower()
    if ans == 'y':
        print(f"[INFO] Confirmed correct => would click on '{target_class}' now.")
        # you can do pyautogui.click() if desired
    else:
        # user correct => compute new offset
        print("[INFO] Move mouse to correct location and press Enter ...")
        input()
        corrected_pos = pyautogui.position()
        dx = corrected_pos.x - cx
        dy = corrected_pos.y - cy
        offsets[target_class] = {"x": dx, "y": dy}
        save_offsets(offsets)
        print(f"[INFO] Offset updated => dx={dx}, dy={dy}.")

        # shift bounding box => treat as new label => store in val
        shifted_x1 = x1 + dx
        shifted_x2 = x2 + dx
        shifted_y1 = y1 + dy
        shifted_y2 = y2 + dy
        save_label_in_val(screenshot_cv, shifted_x1, shifted_y1, shifted_x2, shifted_y2)
        print("[INFO] Doing short train pass with newly corrected val label.")
        short_train(best_path, epochs=1)

##################################################
# USER LABEL (Val Folder)
##################################################
drawing = False
sx, sy, ex, ey = -1, -1, -1, -1

def mouse_callback(event, x, y, flags, param):
    global drawing, sx, sy, ex, ey
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y
        ex, ey = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y

def user_label_val(image_cv, target_class_id=0):
    """
    Opens an OpenCV window => user draws bounding box => press 's' to save,
    'r' to reset, or 'q' to skip. Then saves to data/images/val + data/labels/val
    """
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    wname = "Label VAL - draw box, press 's' to save, 'r' to reset, 'q' to skip"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, mouse_callback)

    global sx, sy, ex, ey
    sx = sy = ex = ey = -1

    clone = image_cv.copy()
    while True:
        temp = clone.copy()
        if sx != -1 and sy != -1 and ex != -1 and ey != -1:
            cv2.rectangle(temp, (sx, sy), (ex, ey), (0,255,0), 2)

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
            img_name = f"val_user_{tstamp}.jpg"
            img_path = os.path.join(VAL_IMG_DIR, img_name)
            cv2.imwrite(img_path, image_cv)
            print(f"[INFO] Saved VAL image => {img_path}")

            # YOLO label
            h, w = image_cv.shape[:2]
            xx1, xx2 = sorted([sx, ex])
            yy1, yy2 = sorted([sy, ey])
            bw = xx2 - xx1
            bh = yy2 - yy1
            x_center = (xx1 + bw/2)/w
            y_center = (yy1 + bh/2)/h
            norm_w = bw/w
            norm_h = bh/h

            label_path = os.path.join(VAL_LABEL_DIR, img_name.replace(".jpg",".txt"))
            with open(label_path, "w") as lf:
                line = f"{target_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line + "\n")

            print(f"[INFO] Saved VAL label => {label_path}")
            print(f"[DEBUG] Label contents => {line}")
            cv2.destroyWindow(wname)
            return True

        elif key == ord('q'):
            print("[INFO] Skipped labeling => no new data in val.")
            cv2.destroyWindow(wname)
            return False

def save_label_in_val(image_cv, x1, y1, x2, y2, class_id=0):
    """
    Similar logic to user_label_val, but for a bounding box we already have (shifted).
    """
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    tstamp = int(time.time())
    img_name = f"val_corrected_{tstamp}.jpg"
    img_path = os.path.join(VAL_IMG_DIR, img_name)
    cv2.imwrite(img_path, image_cv)

    h, w = image_cv.shape[:2]
    xx1, xx2 = sorted([x1, x2])
    yy1, yy2 = sorted([y1, y2])
    bw = xx2 - xx1
    bh = yy2 - yy1
    x_center = (xx1 + bw/2)/w
    y_center = (yy1 + bh/2)/h
    norm_w = bw/w
    norm_h = bh/h

    label_path = os.path.join(VAL_LABEL_DIR, img_name.replace(".jpg",".txt"))
    line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
    with open(label_path, "w") as lf:
        lf.write(line + "\n")

    print(f"[INFO] Saved VAL corrected image => {img_path}")
    print(f"[INFO] Saved VAL label => {label_path}")
    print(f"[DEBUG] Label contents => {line}")

##################################################
# MAIN
##################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="PlateUp!", help="Which class to detect/correct.")
    args = parser.parse_args()

    while True:
        detect_and_label(args.target)
        cont = input("Press Enter to run again, or 'q' to quit: ").lower()
        if cont == 'q':
            print("[INFO] Exiting.")
            break

if __name__ == "__main__":
    main()
