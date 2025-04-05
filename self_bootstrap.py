import os
import time
import argparse
import cv2
import numpy as np

import pyautogui
from pywinauto import Desktop
from ultralytics import YOLO

##################################################
# CONFIG
##################################################
DATA_YAML = "data.yaml"          # Path to your train/val sets
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"  # Final YOLO model we read/write
LAST_PT = os.path.join(MODEL_OUTPUT_DIR, "weights", "last.pt")

# Where newly labeled images go
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"

VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"

CONF_THRESHOLD = 0.5  # high confidence for auto-label
EPOCHS_QUICK = 1       # short train passes

##################################################
# MAKE DIRS
##################################################
def ensure_dirs():
    """
    Create data/images/train, data/labels/train,
    data/images/val, data/labels/val if needed.
    """
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

##################################################
# TRAIN / RESUME
##################################################
def train_or_resume(base_model):
    """
    Attempt to resume from last.pt for a quick pass.
    If there's "nothing to resume," do a short fresh train.
    If last.pt doesn't exist at all, train from base_model from scratch.
    """
    if os.path.exists(LAST_PT):
        print("[INFO] Attempting a quick resume from last.pt for 1 epoch.")
        model = YOLO(LAST_PT)
        try:
            model.train(
                data=DATA_YAML,
                epochs=300,
                imgsz=1920,
                batch=5,
                project="runs/detect",
                name="train",
                exist_ok=True,
                resume=True
            )
        except AssertionError as e:
            # e.g. "nothing to resume" => do a fresh train from last.pt
            print(f"[WARN] Caught assertion error: {e}")
            print("[WARN] Doing a 'fresh' train from last.pt instead.")
            model = YOLO(LAST_PT)
            model.train(
                data=DATA_YAML,
                epochs=300,
                imgsz=1920,
                batch=5,
                project="runs/detect",
                name="train",
                exist_ok=True,
                resume=False  # fresh short pass
            )
    else:
        # fallback to base_model (best.pt or yolov8s.pt if best doesn't exist)
        print(f"[INFO] No last.pt found, training from: {base_model}")
        model = YOLO(base_model)
        model.train(
            data=DATA_YAML,
            epochs=300,   # a bit more for the first time
            imgsz=1920,
            batch=5,
            project="runs/detect",
            name="train",
            exist_ok=True
        )
    print("[INFO] Training pass done. Check runs/detect/train/weights.")

##################################################
# CAPTURE SCREEN (FOCUS A WINDOW FIRST)
##################################################
def capture_window(window_title="Steam"):
    """
    1) Attempt to find a window by title regex (default: ".*Steam.*").
    2) Bring that window to the foreground (set focus).
    3) Then capture a full-screen screenshot using pyautogui.
    """
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        # Bring it to the front
        win.set_focus()
        time.sleep(0.5)  # brief pause to ensure focus
        print(f"[INFO] Focused window matching '{window_title}'.")
    except Exception as e:
        print(f"[WARN] Could not focus window '{window_title}': {e}")

    # Always capture entire screen after focusing
    screenshot = pyautogui.screenshot()
    image_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("[INFO] Captured full-screen screenshot.")
    return image_cv

##################################################
# SHOW DETECTIONS & MOVE MOUSE
##################################################
def show_detections_and_move_mouse(image_cv, results, conf_threshold=CONF_THRESHOLD):
    """
    1) Draw bounding boxes (above threshold) on a copy of the image.
    2) Move mouse to the center of the highest-confidence box (for demonstration).
    3) Show the image in a small CV2 window briefly so user sees where the box is.
    4) Return the same results object for further processing (auto-label, etc.).
    """
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return results

    display_img = image_cv.copy()
    max_conf = -1
    best_center = (None, None)

    for box in results.boxes:
        conf = box.conf[0].item()
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color=(0, 255, 0), thickness=2)
        if conf > max_conf:
            max_conf = conf
            best_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))

    # If we found a best box, move the mouse there
    if max_conf >= 0 and best_center != (None, None):
        cx, cy = best_center
        print(f"[INFO] Moving mouse to best box center: ({cx}, {cy}), conf={max_conf:.3f}")
        pyautogui.moveTo(cx, cy, duration=1)

    # Show detection results for ~2 seconds
    cv2.imshow("Detections (Press any key to continue)", display_img)
    cv2.waitKey(2000)
    cv2.destroyWindow("Detections (Press any key to continue)")

    return results

##################################################
# AUTO-LABEL (GOES TO TRAIN FOLDER)
##################################################
def auto_label_image(image_cv, results):
    """
    Auto-label all boxes with conf >= CONF_THRESHOLD into the train folder.
    Returns number of boxes auto-labeled.
    """
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return 0

    tstamp = int(time.time())
    img_name = f"auto_{tstamp}.jpg"
    img_path = os.path.join(TRAIN_IMG_DIR, img_name)
    cv2.imwrite(img_path, image_cv)

    label_path = os.path.join(TRAIN_LABEL_DIR, img_name.replace(".jpg", ".txt"))
    lines = []
    h, w = image_cv.shape[:2]
    count_labeled = 0

    for box in results.boxes:
        conf = box.conf[0].item()
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0].item())  # YOLO's predicted class
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        bw = x2 - x1
        bh = y2 - y1
        x_center = (x1 + bw / 2) / w
        y_center = (y1 + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        lines.append(line)
        count_labeled += 1
        print(f"  [AUTO] cls={cls_id} conf={conf:.2f} => {line}")

    # If no bounding boxes remain above threshold, remove the image
    if count_labeled == 0:
        os.remove(img_path)
        return 0

    with open(label_path, 'w') as lf:
        for l in lines:
            lf.write(l + "\n")

    print(f"[INFO] Auto-labeled => {img_path}")
    print(f"[INFO] Created label => {label_path} with {count_labeled} boxes.")
    return count_labeled

##################################################
# USER LABEL
##################################################
drawing = False
sx, sy, ex, ey = -1, -1, -1, -1

def mouse_cb(event, x, y, flags, param):
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

def train_folder_is_empty():
    """Returns True if data/images/train has no .jpg/.png/etc. files."""
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for f in os.listdir(TRAIN_IMG_DIR):
        if f.lower().endswith(valid_exts):
            return False
    return True

def user_label_image(image_cv, class_id=0, use_val=True):
    """
    Opens an OpenCV window => user draws a bounding box => press 's' to save or 'q' to skip.
    If use_val=False (or train folder is empty), we place it in the train folder.
    Otherwise, we place it in the val folder.
    
    class_id: integer class ID for the bounding box (0=PlateUp!, 1=Factorio, etc.)
    """
    global sx, sy, ex, ey, drawing
    wname = "User Label - draw box (s=save, q=skip)"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, mouse_cb)

    sx = sy = ex = ey = -1

    clone = image_cv.copy()
    while True:
        temp = clone.copy()
        if sx != -1 and sy != -1 and ex != -1 and ey != -1:
            cv2.rectangle(temp, (sx, sy), (ex, ey), (0, 255, 0), 2)

        cv2.imshow(wname, temp)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            # User pressed 's' => save the bounding box
            if sx == ex or sy == ey:
                print("[WARN] No bounding box drawn.")
                continue

            # Ensure the first labeled image always goes to train if empty
            if train_folder_is_empty():
                use_val = False

            if not use_val:
                # put in train set
                img_dir = TRAIN_IMG_DIR
                label_dir = TRAIN_LABEL_DIR
            else:
                # put in val set
                img_dir = VAL_IMG_DIR
                label_dir = VAL_LABEL_DIR

            tstamp = int(time.time())
            fname = f"user_{tstamp}.jpg"
            img_path = os.path.join(img_dir, fname)
            cv2.imwrite(img_path, image_cv)

            h, w = image_cv.shape[:2]
            xx1, xx2 = sorted([sx, ex])
            yy1, yy2 = sorted([sy, ey])
            bw = xx2 - xx1
            bh = yy2 - yy1
            x_center = (xx1 + bw/2) / w
            y_center = (yy1 + bh/2) / h
            norm_w = bw / w
            norm_h = bh / h

            lbl_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))
            with open(lbl_path, 'w') as lf:
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line + "\n")

            print(f"[INFO] Saved user-labeled => {img_path}")
            print(f"[INFO] Created label => {lbl_path}")
            with open(lbl_path, 'r') as f:
                print("Label contents:\n" + f.read())

            cv2.destroyWindow(wname)
            return True

        elif key == ord('q'):
            print("[INFO] Skipped user labeling => no label created.")
            cv2.destroyWindow(wname)
            return False

##################################################
# SELF-TRAIN LOOP (RUNS FOREVER)
##################################################
def self_train_loop(window_title, default_class_id):
    """
    Infinite loop:
      1) Bring the target window to the front (via pywinauto) 
         and capture the full screen with pyautogui.
      2) YOLO detection
      3) Show bounding boxes, possibly move mouse
      4) Auto-label to TRAIN or prompt user label to VAL
         (unless train is empty => user-labeled goes to TRAIN)
      5) Short training pass
      6) Repeat
    """
    ensure_dirs()

    # Base model
    base_model = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    if not os.path.exists(base_model):
        base_model = "yolov8s.pt"

    print("\n[INFO] ENTERING ENDLESS SELF-TRAIN LOOP.\n"
          "      Press Ctrl+C in the console to stop.\n")

    while True:
        # 1) Capture full screen after focusing a window (e.g. Steam)
        image_cv = capture_window(window_title)

        # 2) Detect with YOLO
        model = YOLO(base_model)
        res_list = model.predict(source=image_cv, imgsz=1920)
        if not res_list:
            print("[WARN] No inference results. Asking for user labeling (check if train is empty).")
            user_label_image(image_cv, class_id=default_class_id, use_val=True)
            train_or_resume(base_model)
            continue

        results = res_list[0]

        # 3) Show bounding boxes & move mouse
        results = show_detections_and_move_mouse(image_cv, results, CONF_THRESHOLD)

        # 4) Auto-label or user-labeled
        count_labeled = auto_label_image(image_cv, results)
        if count_labeled == 0:
            print("[INFO] No boxes above conf threshold => user labeling (val).")
            user_label_image(image_cv, class_id=default_class_id, use_val=True)

        # 5) Short train pass
        train_or_resume(base_model)

        # 6) Loop forever until Ctrl+C

##################################################
# MAIN
##################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=str, default="Steam",
                        help="Window title (or partial) to focus before screenshot.")
    parser.add_argument("--labelclass", type=int, default=0,
                        help="Class ID for manual labeling: 0=PlateUp!, 1=Factorio, etc.")
    args = parser.parse_args()

    # Start the loop using the specified window title and class ID
    self_train_loop(window_title=args.window, default_class_id=args.labelclass)

if __name__ == "__main__":
    main()
