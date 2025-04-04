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
DATA_YAML = "data.yaml"  # path to your YOLO dataset config
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"  # YOLO saves best.pt in 'weights/' by default
OFFSET_FILE = "mouse_offsets.json"

# By default, we store on-the-fly corrections in data/images/train & data/labels/train
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"


##################################################
# OFFSET HANDLING
##################################################
def load_offsets():
    """Load offsets from JSON (used if user corrects the mouse position)."""
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not parse offsets. Returning empty dict.")
    return {}

def save_offsets(offsets):
    """Save offsets to JSON."""
    with open(OFFSET_FILE, "w") as f:
        json.dump(offsets, f, indent=4)


##################################################
# TRAIN OR RESUME MODEL
##################################################
def train_or_resume_model(resume=False):
    """
    Trains (or resumes) a YOLO model.
    If resume=True, we'll use 'last.pt' from runs/detect/train/weights.
    Otherwise, we start from 'yolov8s.pt' pretrained weights.
    """
    model_path = os.path.join(MODEL_OUTPUT_DIR, "weights", "last.pt")

    if resume and os.path.exists(model_path):
        print("[INFO] Resuming model from last checkpoint...")
        model = YOLO(model_path)
        model.train(
            data=DATA_YAML,
            epochs=1,   # Just do 1 epoch for quick online updates
            batch=4,
            imgsz=640,
            project="runs/detect",
            name="train",
            exist_ok=True,
            verbose=True,
            deterministic=True,
            resume=True,
        )
    else:
        print("[INFO] Starting fresh training (no best.pt found).")
        model = YOLO("yolov8s.pt")
        model.train(
            data=DATA_YAML,
            epochs=5,
            batch=4,
            imgsz=640,
            project="runs/detect",
            name="train",
            exist_ok=True,
            verbose=True,
            deterministic=True,
        )
    print("[INFO] Training (or resume) complete. Check runs/detect/train/weights for results.")


##################################################
# DETECT & PROMPT USER
##################################################
def detect_and_learn(target_class: str = "PlateUp!"):
    offsets = load_offsets()
    offset_x = offsets.get(target_class, {}).get("x", 0)
    offset_y = offsets.get(target_class, {}).get("y", 0)

    # 1) Ensure model weights exist
    best_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    last_path = os.path.join(MODEL_OUTPUT_DIR, "weights", "last.pt")

    # If no best.pt or last.pt, we can't proceed. We'll do a fresh training
    if not os.path.exists(best_path) and not os.path.exists(last_path):
        print("[WARN] No weights found. Training from scratch.")
        train_or_resume_model(resume=False)

    # 2) Load the YOLO model
    model_path = best_path if os.path.exists(best_path) else last_path
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)

    # 3) Capture screenshot
    screenshot = pyautogui.screenshot()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 4) Run detection
    results = model(screenshot_cv)[0]
    found_box = None
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        class_name = results.names[cls_id]
        if class_name == target_class:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            found_box = (x1, y1, x2, y2)
            break

    if not found_box:
        print(f"[INFO] No bounding box found for '{target_class}'.")
        # We'll still let the user label it manually if desired
        manual_improve_data(screenshot_cv, target_class)
        # Then re-train quickly (resume)
        print("[INFO] Re-training after user labeling (since detection failed).")
        train_or_resume_model(resume=True)
        return

    # 5) If found, move the mouse to bounding box center + offset
    x1, y1, x2, y2 = found_box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    print(f"[INFO] Detected '{target_class}' at center: ({center_x}, {center_y}).")

    final_x = center_x + offset_x
    final_y = center_y + offset_y
    pyautogui.moveTo(final_x, final_y, duration=1)
    print(f"[INFO] Mouse moved to: ({final_x}, {final_y}).")

    # 6) Ask user if correct
    feedback = input("Is the mouse on the correct spot? (y/n): ").lower()
    if feedback == 'y':
        # Great, we just click
        pyautogui.click()
        print(f"[INFO] Clicked on '{target_class}'.")
    else:
        # Let user correct bounding box OR just correct final mouse coords
        choice = input("Type 'mouse' to just correct the mouse position, or 'box' to re-draw bounding box: ").lower()
        if choice == 'mouse':
            # Just correct final pointer location
            print("[INFO] Move mouse to the correct location and press Enter...")
            input()
            corrected_pos = pyautogui.position()
            offset_x = corrected_pos.x - center_x
            offset_y = corrected_pos.y - center_y
            offsets[target_class] = {"x": offset_x, "y": offset_y}
            save_offsets(offsets)
            print(f"[INFO] Offset for '{target_class}' updated to: ({offset_x}, {offset_y}).")

            # Move again and click
            pyautogui.moveTo(corrected_pos.x, corrected_pos.y, duration=1)
            pyautogui.click()
            print(f"[INFO] Clicked on '{target_class}'.")

        elif choice == 'box':
            # We'll re-draw the bounding box on this screenshot
            manual_improve_data(screenshot_cv, target_class)
            print("[INFO] Re-training after user re-draw (resume).")
            train_or_resume_model(resume=True)


##################################################
# MANUAL BOX DRAWING & DATA STORAGE
##################################################
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, ex, ey
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ex, ey = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y

def manual_improve_data(image_cv, class_name="PlateUp!"):
    """
    Lets the user draw a bounding box on the current screenshot, saves that image & YOLO label
    to data/images/train & data/labels/train, so we can train again with the new data.
    """

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

    clone = image_cv.copy()
    win_name = "Labeling - Draw bounding box, press 's' to save, 'q' to quit"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    global ix, iy, ex, ey
    ix = iy = ex = ey = -1

    while True:
        temp = clone.copy()
        if ix != -1 and iy != -1 and ex != -1 and ey != -1:
            cv2.rectangle(temp, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.imshow(win_name, temp)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            if ix == ex or iy == ey:
                print("[WARN] No bounding box drawn. Try again.")
                continue

            # Save the screenshot
            filename = f"correction_{int(time.time())}.jpg"
            image_path = os.path.join(TRAIN_IMG_DIR, filename)
            cv2.imwrite(image_path, image_cv)

            # YOLO label
            class_id = 0  # single class scenario
            h, w = image_cv.shape[:2]
            x1, x2 = sorted([ix, ex])
            y1, y2 = sorted([iy, ey])
            box_w = x2 - x1
            box_h = y2 - y1
            x_center = (x1 + box_w / 2) / w
            y_center = (y1 + box_h / 2) / h
            box_w_rel = box_w / w
            box_h_rel = box_h / h

            label_path = os.path.join(TRAIN_LABEL_DIR, filename.replace(".jpg", ".txt"))
            with open(label_path, 'w') as lf:
                lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                         f"{box_w_rel:.6f} {box_h_rel:.6f}\n")

            print(f"[INFO] Saved corrected image => {image_path}")
            print(f"[INFO] Saved label => {label_path}")
            break

        elif key == ord('q'):
            print("[INFO] Canceling bounding box correction.")
            break

    cv2.destroyWindow(win_name)


##################################################
# MAIN
##################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Perform full training before detection.")
    parser.add_argument("--target", type=str, default="PlateUp!", help="Which class to detect/correct.")
    args = parser.parse_args()

    # If user wants a fresh training or we have no best.pt, do it
    best_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    last_path = os.path.join(MODEL_OUTPUT_DIR, "weights", "last.pt")
    if args.train or (not os.path.exists(best_path) and not os.path.exists(last_path)):
        train_or_resume_model(resume=False)

    # Then detection + user feedback
    detect_and_learn(args.target)


if __name__ == "__main__":
    main()
