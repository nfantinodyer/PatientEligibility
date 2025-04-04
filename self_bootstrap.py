import os
import time
import argparse
import cv2
import numpy as np

import pyautogui
from pywinauto import Desktop  # <-- for capturing window screenshots
from ultralytics import YOLO

##################################################
# CONFIG
##################################################
DATA_YAML = "data.yaml"  # path to your train/val sets
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"       # final YOLO model we read/write
LAST_PT = os.path.join(MODEL_OUTPUT_DIR, "weights", "last.pt")

# Where newly labeled images go for training
USER_IMG_DIR = "data/images/train"
USER_LABEL_DIR = "data/labels/train"

CONF_THRESHOLD = 0.90  # high conf for auto-label
EPOCHS_QUICK = 1       # how many epochs each “short train” does

##################################################
# MAKE DIRS
##################################################
def ensure_dirs():
    os.makedirs(USER_IMG_DIR, exist_ok=True)
    os.makedirs(USER_LABEL_DIR, exist_ok=True)

##################################################
# TRAIN / RESUME
##################################################
def train_or_resume(base_model):
    """
    Try resuming from last.pt for a short pass. If "nothing to resume",
    do a fresh short train from last.pt as if it's pretrained weights.
    If last.pt doesn't exist, train from base_model from scratch.
    """
    if os.path.exists(LAST_PT):
        print("[INFO] Attempting a quick resume from last.pt for 1 epoch.")
        model = YOLO(LAST_PT)
        try:
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS_QUICK,
                imgsz=640,
                batch=4,
                project="runs/detect",
                name="train",
                exist_ok=True,
                resume=True
            )
        except AssertionError as e:
            # e.g. "nothing to resume" => do a fresh short train from last.pt
            print(f"[WARN] Caught assertion error: {e}")
            print("[WARN] Doing a 'fresh' train from last.pt instead.")
            model = YOLO(LAST_PT)
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS_QUICK,
                imgsz=640,
                batch=4,
                project="runs/detect",
                name="train",
                exist_ok=True,
                resume=False
            )
    else:
        # fallback to base_model (best.pt or yolov8s.pt if best does not exist)
        print(f"[INFO] No last.pt found, training from: {base_model}")
        model = YOLO(base_model)
        model.train(
            data=DATA_YAML,
            epochs=5,   # a bit more for the first time
            imgsz=640,
            batch=4,
            project="runs/detect",
            name="train",
            exist_ok=True
        )
    print("[INFO] Training pass done. Check runs/detect/train/weights.")

##################################################
# CAPTURE WINDOW VIA PYWINAUTO
##################################################
def capture_window(window_title="PlateUp!"):
    """
    Attempt to locate a window by title via pywinauto Desktop,
    then capture it as a Pillow image, then convert to cv2 (numpy) BGR format.
    Fallback: if not found, does a pyautogui full-screen screenshot.
    """
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        pil_img = win.capture_as_image()  # returns a PIL Image
        if pil_img is None:
            raise ValueError("capture_as_image() returned None.")
        image_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print(f"[INFO] Captured window with title containing '{window_title}'.")
        return image_cv
    except Exception as e:
        print(f"[WARN] Could not capture '{window_title}' window: {e}")
        print("[WARN] Falling back to full-screen screenshot with pyautogui.")
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

##################################################
# AUTO-LABEL
##################################################
def auto_label_image(image_cv, results):
    """
    Auto-label all boxes with conf >= CONF_THRESHOLD into data/images/train & data/labels/train.
    Returns number of boxes auto-labeled.
    """
    # results can have .boxes or .xywhn, depending on ultralytics version
    # We'll standardize by reading .boxes
    if not hasattr(results, 'boxes'):
        # older versions store in results[0].boxes
        # but the new usage is results.boxes for each result
        return 0

    boxes = results.boxes
    if len(boxes) == 0:
        return 0

    # Save the image
    tstamp = int(time.time())
    img_name = f"auto_{tstamp}.jpg"
    img_path = os.path.join(USER_IMG_DIR, img_name)
    cv2.imwrite(img_path, image_cv)

    label_path = os.path.join(USER_LABEL_DIR, img_name.replace(".jpg", ".txt"))
    lines = []

    h, w = image_cv.shape[:2]
    count_labeled = 0

    for box in boxes:
        conf = box.conf[0].item()
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0].item())
        # box.xywhn is sometimes [x_center, y_center, width, height] in normalized coords
        # box.xywh is absolute coords. Let’s be consistent with YOLO normalized format
        # The box object has .xywhn or .xyxy. We'll do a direct approach:
        # safer: extract from box.data
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        x_center = (x1 + bw/2) / w
        y_center = (y1 + bh/2) / h
        norm_w = bw / w
        norm_h = bh / h

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        lines.append(line)
        count_labeled += 1

        print(f"  [AUTO] cls={cls_id} conf={conf:.2f} => {line}")

    if count_labeled == 0:
        # remove the image if we wrote it but have no bounding boxes
        os.remove(img_path)
        return 0

    with open(label_path, 'w') as lf:
        for l in lines:
            lf.write(l + "\n")

    print(f"[INFO] Auto-labeled => {img_path}")
    print(f"[INFO] Created label file => {label_path} with {count_labeled} box(es).")
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

def user_label_image(image_cv, class_id=0):
    """
    Opens an OpenCV window => user draws one bounding box => press 's' to save or 'q' to skip.
    """
    wname = "User Label - draw box, s=save, q=skip"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, mouse_cb)

    global sx, sy, ex, ey
    sx = sy = ex = ey = -1

    clone = image_cv.copy()
    while True:
        temp = clone.copy()
        if sx != -1 and sy != -1 and ex != -1 and ey != -1:
            cv2.rectangle(temp, (sx, sy), (ex, ey), (0, 255, 0), 2)

        cv2.imshow(wname, temp)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            if sx == ex or sy == ey:
                print("[WARN] No bounding box drawn.")
                continue

            # Save the new training image
            tstamp = int(time.time())
            fname = f"user_{tstamp}.jpg"
            img_path = os.path.join(USER_IMG_DIR, fname)
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

            lbl_path = os.path.join(USER_LABEL_DIR, fname.replace(".jpg", ".txt"))
            with open(lbl_path, 'w') as lf:
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line + "\n")

            print(f"[INFO] Saved user-labeled => {img_path}")
            print(f"[INFO] Created label => {lbl_path}")
            # Show label contents
            with open(lbl_path, 'r') as f:
                print("Label contents:\n" + f.read())

            cv2.destroyWindow(wname)
            return True

        elif key == ord('q'):
            print("[INFO] Skipped user labeling => no label file created.")
            cv2.destroyWindow(wname)
            return False

##################################################
# SELF-TRAIN LOOP
##################################################
def self_train_loop(window_title="PlateUp!"):
    """
    Continuously:
     1) Capture the specified window (or fallback entire screen)
     2) Run detection
     3) If any detection >= CONF_THRESHOLD, auto-label
        Else, ask user to label via bounding box
     4) Perform a short training pass
     5) Repeat or quit
    """
    # Make sure train directories exist
    ensure_dirs()

    # Base model: either your best.pt or fallback to yolov8s.pt
    base_model = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
    if not os.path.exists(base_model):
        base_model = "yolov8s.pt"

    while True:
        print("\n[INFO] Starting self-training cycle ...")
        print(f"[INFO] Using model => {base_model}")

        # 1) Capture window screenshot
        image_cv = capture_window(window_title)

        # 2) Detect with YOLO
        model = YOLO(base_model)
        # The simplest way is to run inference on the in-memory image:
        # model() can accept np.ndarray directly
        # but note: “model.predict()” or just “model()” depends on your ultralytics version
        res_list = model.predict(source=image_cv, imgsz=640)
        if not res_list:
            print("[WARN] No inference results were returned, skipping.")
            # Possibly ask user to label anyway
            user_label_image(image_cv, class_id=0)
            train_or_resume(base_model)
        else:
            # usually 1 result for 1 image
            results = res_list[0]
            # 3) Attempt auto-label
            count_labeled = auto_label_image(image_cv, results)

            if count_labeled == 0:
                # Not sure => ask user to label
                print("[INFO] No boxes above conf threshold => asking user to label.")
                user_label_image(image_cv, class_id=0)

            # 4) Then short train pass
            train_or_resume(base_model)

        # 5) Ask user if we should loop or quit
        ans = input("Press Enter to capture again, or 'q' to quit: ").lower()
        if ans == 'q':
            print("[INFO] Exiting self-train loop.")
            break

##################################################
# MAIN
##################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=str, default="PlateUp!",
                        help="Window title (or partial) to screenshot with pywinauto")
    args = parser.parse_args()

    self_train_loop(window_title=args.window)

if __name__ == "__main__":
    main()
