import os
import time
import shutil
import argparse
import cv2
import numpy as np

import pyautogui
from pywinauto import Desktop
from ultralytics import YOLO

##################################################
# CONFIG
##################################################
DATA_YAML = "data.yaml"   # Path to your train/val sets
MODEL_OUTPUT_DIR = "runs/detect/train"
MODEL_WEIGHTS = "weights/best.pt"    # Your YOLO final weights
BEST_PT = os.path.join(MODEL_OUTPUT_DIR, "weights", "best.pt")

# Where newly labeled images go
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"

CONF_THRESHOLD = 0.5
SHORT_EPOCHS = 55

##################################################
# MAKE DIRS
##################################################
def ensure_dirs():
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)
    # also ensure runs/detect/train/weights exists if you want
    os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "weights"), exist_ok=True)

##################################################
# SHORT TRAIN (ALWAYS FRESH) in a UNIQUE SUBFOLDER,
# THEN COPY best.pt => MAIN BEST
##################################################
def short_fresh_train():
    """
    1) Pick your base model (BEST_PT if it exists, else fallback).
    2) Train to a unique subfolder (timestamp-based).
    3) Copy the newly trained best.pt => runs/detect/train/weights/best.pt
    This avoids Windows OSError: [Errno 22] by never re-writing the same last.pt in the same run.
    """
    from ultralytics import YOLO

    # 1) base weights
    if os.path.exists(BEST_PT):
        base = BEST_PT
        print(f"[INFO] Short fresh train from {BEST_PT}")
    else:
        fallback = os.path.join(MODEL_OUTPUT_DIR, MODEL_WEIGHTS)
        if os.path.exists(fallback):
            base = fallback
            print(f"[INFO] Short fresh train from {fallback}")
        else:
            base = "yolov8s.pt"
            print("[INFO] Short fresh train from yolov8s.pt (no local best.pt)")

    # 2) unique subfolder => 'train_1691261234'
    timestamp = int(time.time())
    train_subfolder = f"train_{timestamp}"
    subdir_path = f"runs/detect/{train_subfolder}"

    model = YOLO(base)
    model.train(
        data=DATA_YAML,
        epochs=SHORT_EPOCHS,
        imgsz=1920,
        batch=7,
        project="runs/detect",
        name=train_subfolder,   # put results in 'runs/detect/train_TIMESTAMP/'
        exist_ok=True,
        resume=False,
    )
    print(f"[INFO] Short fresh training done. Check runs/detect/{train_subfolder}/weights/")

    # 3) copy best => main best
    new_best_path = f"runs/detect/{train_subfolder}/weights/best.pt"
    if os.path.exists(new_best_path):
        shutil.copyfile(new_best_path, BEST_PT)
        print(f"[INFO] Copied {new_best_path} => {BEST_PT}")
        shutil.rmtree( subdir_path )
    else:
        print(f"[WARN] Did not find {new_best_path} => skipping copy to main best {BEST_PT}")

##################################################
# CAPTURE SCREEN
##################################################
def capture_window(window_title="Steam"):
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        win.set_focus()
        time.sleep(0.5)
        print(f"[INFO] Focused window matching '{window_title}'.")
    except Exception as e:
        print(f"[WARN] Could not focus window '{window_title}': {e}")

    screenshot = pyautogui.screenshot()
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("[INFO] Captured full-screen screenshot.")
    return img_cv

##################################################
# DETECTION / MOUSE
##################################################
def show_detections_and_move_mouse(image_cv, results, CONF_THRESHOLD, default_class_id):
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return results

    disp_img = image_cv.copy()
    max_conf = -1
    best_center = (None, None)

    for box in results.boxes:
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        if conf < CONF_THRESHOLD or cls_id != default_class_id:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cv2.rectangle(disp_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        if conf > max_conf:
            max_conf = conf
            best_center = (int((x1+x2)//2), int((y1+y2)//2))

    if max_conf >= 0 and best_center != (None, None):
        cx, cy = best_center
        print(f"[INFO] Move mouse => cls={default_class_id}, center=({cx},{cy}), conf={max_conf:.3f}")
        pyautogui.moveTo(cx, cy, duration=1)
    else:
        print(f"[INFO] No detection for class_id={default_class_id} above threshold.")

    wname = "Detections (Press any key to continue)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, 2560, 1440)
    try:
        cv2.setWindowProperty(wname, cv2.WND_PROP_TOPMOST, 1)
    except Exception as e:
        print(f"[WARN] Could not topmost: {e}")

    cv2.imshow(wname, disp_img)
    cv2.waitKey(2000)
    cv2.destroyWindow(wname)
    return results

##################################################
# AUTO-LABEL
##################################################
def auto_label_image(image_cv, results, default_class_id):
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return 0

    tstamp = int(time.time())
    fname = f"auto_{tstamp}.jpg"
    img_path = os.path.join(TRAIN_IMG_DIR, fname)
    cv2.imwrite(img_path, image_cv)

    label_path = os.path.join(TRAIN_LABEL_DIR, fname.replace(".jpg", ".txt"))
    lines = []
    h, w = image_cv.shape[:2]
    count = 0

    for box in results.boxes:
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        if conf < CONF_THRESHOLD or cls_id != default_class_id:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        bw = x2 - x1
        bh = y2 - y1
        x_center = (x1 + bw/2)/w
        y_center = (y1 + bh/2)/h
        norm_w = bw/w
        norm_h = bh/h

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        lines.append(line)
        print(f"[AUTO] cls={cls_id}, conf={conf:.2f} => {line}")
        count += 1

    if count == 0:
        os.remove(img_path)
        return 0

    with open(label_path, 'w') as f:
        for l in lines:
            f.write(l+"\n")

    print(f"[INFO] Auto-labeled => {img_path}")
    print(f"[INFO] Created label => {label_path} with {count} boxes.")
    return count

##################################################
# USER LABEL
##################################################
drawing = False
sx=sy=ex=ey = -1

def mouse_cb(event, x, y, flags, param):
    global drawing,sx,sy,ex,ey
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx,sy = x,y
        ex,ey = x,y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex,ey = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex,ey = x,y

def train_folder_is_empty():
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return not any(f.lower().endswith(exts) for f in os.listdir(TRAIN_IMG_DIR))

def user_label_image(image_cv, class_id, use_val=True):
    global sx,sy,ex,ey,drawing
    wname="User Label - draw box (s=save, q=skip)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname,2560,1440)
    try:
        cv2.setWindowProperty(wname, cv2.WND_PROP_TOPMOST,1)
    except:
        pass
    cv2.setMouseCallback(wname,mouse_cb)

    sx=sy=ex=ey=-1
    clone = image_cv.copy()

    while True:
        temp = clone.copy()
        if sx!=-1 and sy!=-1 and ex!=-1 and ey!=-1:
            cv2.rectangle(temp,(sx,sy),(ex,ey),(0,255,0),2)

        cv2.imshow(wname,temp)
        k = cv2.waitKey(10)&0xFF

        if k==ord('s'):
            if sx==ex or sy==ey:
                print("[WARN] No bounding box drawn.")
                continue

            if train_folder_is_empty():
                use_val=False

            if not use_val:
                img_dir=TRAIN_IMG_DIR
                label_dir=TRAIN_LABEL_DIR
            else:
                img_dir=VAL_IMG_DIR
                label_dir=VAL_LABEL_DIR

            tstamp = int(time.time())
            fname = f"user_{tstamp}.jpg"
            img_path=os.path.join(img_dir,fname)
            cv2.imwrite(img_path,image_cv)

            h,w=image_cv.shape[:2]
            xx1,xx2=sorted([sx,ex])
            yy1,yy2=sorted([sy,ey])
            bw=xx2-xx1
            bh=yy2-yy1
            x_center=(xx1+bw/2)/w
            y_center=(yy1+bh/2)/h
            norm_w=bw/w
            norm_h=bh/h

            lbl_path=os.path.join(label_dir,fname.replace(".jpg",".txt"))
            with open(lbl_path,'w') as lf:
                line=f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line+"\n")

            print(f"[INFO] Saved user-labeled => {img_path}")
            print(f"[INFO] Created label => {lbl_path}")
            with open(lbl_path,'r') as f:
                print("Label contents:\n"+f.read())

            cv2.destroyWindow(wname)
            return True

        elif k==ord('q'):
            print("[INFO] Skipped user labeling => no label created.")
            cv2.destroyWindow(wname)
            return False

##################################################
# SELF-TRAIN LOOP
##################################################
def self_train_loop(window_title, default_class_id):
    ensure_dirs()
    print("[INFO] ENTERING ENDLESS SELF-TRAIN LOOP. Ctrl+C to stop.\n")

    while True:
        image_cv = capture_window(window_title)

        # base model
        if os.path.exists(BEST_PT):
            base=BEST_PT
        else:
            fallback=os.path.join(MODEL_OUTPUT_DIR,MODEL_WEIGHTS)
            base=fallback if os.path.exists(fallback) else "yolov8s.pt"

        model=YOLO(base)
        res_list=model.predict(source=image_cv, imgsz=1920)
        if not res_list:
            print("[WARN] No inference => user label.")
            user_label_image(image_cv,class_id=default_class_id,use_val=True)
            short_fresh_train()
            continue

        results=res_list[0]

        show_detections_and_move_mouse(image_cv, results, CONF_THRESHOLD, default_class_id)

        c=auto_label_image(image_cv, results, default_class_id)
        if c==0:
            print("[INFO] No boxes => manual label.")
            user_label_image(image_cv,class_id=default_class_id,use_val=True)

        short_fresh_train()

##################################################
# MAIN
##################################################
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--window",type=str,default="Steam",help="Window to focus.")
    parser.add_argument("--labelclass",type=int,default=0,help="Class ID for detection.")
    #1 for Factorio,0 for PlateUp!
    args=parser.parse_args()

    self_train_loop(window_title=args.window, default_class_id=args.labelclass)

if __name__=="__main__":
    main()
