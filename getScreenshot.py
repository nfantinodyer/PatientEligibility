import os
import time
import argparse
import subprocess
import cv2
import numpy as np

import pyautogui
from pywinauto import Desktop
from ultralytics import YOLO

##################################################
# CONFIG
##################################################
DATA_YAML = "data.yaml"

BEST_MODEL_DIR = "runs/detect/train/weights"
BEST_PT = os.path.join(BEST_MODEL_DIR, "best.pt")

TRAIN_IMG_DIR  = "data/images/train"
TRAIN_LABEL_DIR= "data/labels/train"
VAL_IMG_DIR    = "data/images/val"
VAL_LABEL_DIR  = "data/labels/val"

CONF_THRESHOLD = 0.5

##################################################
# CREATE DIRS
##################################################
def ensure_dirs():
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

##################################################
# CAPTURE WINDOW
##################################################
def capture_window(window_title):
    """
    Focus a window (if found) by partial title, wait 0.5s, 
    then do a full-screen screenshot via pyautogui, 
    returning a BGR OpenCV image.
    """
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        win.set_focus()
        time.sleep(0.5)
        print(f"[INFO] Focused window '{window_title}'.")
    except Exception as e:
        print(f"[WARN] Could not focus window '{window_title}': {e}")

    screenshot = pyautogui.screenshot()
    image_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("[INFO] Captured full-screen screenshot.")
    return image_cv

##################################################
# DETECT CLASS
##################################################
def detect_class(image_cv, class_id=0, conf_thres=CONF_THRESHOLD):
    """
    Load best model if it exists, else yolov8s.pt.
    Detect => see if there's a bounding box with class_id above conf_thres.
    Return True if found, else False.
    """
    model_path = BEST_PT if os.path.exists(BEST_PT) else "yolov8s.pt"
    model = YOLO(model_path)
    results_list = model.predict(source=image_cv, imgsz=1920)
    if not results_list or len(results_list)==0:
        return False

    results = results_list[0]
    if not hasattr(results, 'boxes') or len(results.boxes)==0:
        return False

    for box in results.boxes:
        conf  = box.conf[0].item()
        cls   = int(box.cls[0].item())
        if cls == class_id and conf >= conf_thres:
            return True

    return False

##################################################
# USER LABEL
##################################################
drawing=False
sx=sy=ex=ey=-1

def mouse_cb(event,x,y,flags,param):
    global drawing,sx,sy,ex,ey
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        sx,sy=x,y
        ex,ey=x,y
    elif event==cv2.EVENT_MOUSEMOVE and drawing:
        ex,ey=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        ex,ey=x,y

def user_label_image(image_cv, class_id=0):
    """
    Hard-coded user labeling for a single bounding box => class_id is 0 or 1.
    After user draws => press 's' to save or 'q' to skip.
    Always saves to train folder.
    """
    global sx,sy,ex,ey,drawing
    wname=f"Label Class {class_id} - (s=save, q=skip)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname,1280,720)
    cv2.setMouseCallback(wname, mouse_cb)

    sx=sy=ex=ey=-1
    clone = image_cv.copy()

    while True:
        temp=clone.copy()
        if sx!=-1 and sy!=-1 and ex!=-1 and ey!=-1:
            cv2.rectangle(temp,(sx,sy),(ex,ey),(0,255,0),2)

        cv2.imshow(wname,temp)
        key = cv2.waitKey(10)&0xFF

        if key==ord('s'):
            if sx==ex or sy==ey:
                print("[WARN] No bounding box drawn, try again.")
                continue

            # save to train
            tstamp=int(time.time())
            fname=f"user_{class_id}_{tstamp}.jpg"
            img_path = os.path.join(TRAIN_IMG_DIR,fname)
            cv2.imwrite(img_path,image_cv)

            h,w=image_cv.shape[:2]
            xx1,xx2=sorted([sx,ex])
            yy1,yy2=sorted([sy,ey])
            bw=xx2-xx1
            bh=yy2-yy1
            x_c=(xx1+bw/2)/w
            y_c=(yy1+bh/2)/h
            norm_w=bw/w
            norm_h=bh/h

            lbl_path=os.path.join(TRAIN_LABEL_DIR,fname.replace(".jpg",".txt"))
            with open(lbl_path,"w") as lf:
                line=f"{class_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line+"\n")

            print(f"[INFO] Labeled => {img_path}, class={class_id}")
            print(f"[INFO] Label => {lbl_path}")
            cv2.destroyWindow(wname)
            return True

        elif key==ord('q'):
            print("[INFO] Skipped labeling => no new data.")
            cv2.destroyWindow(wname)
            return False

##################################################
# MAIN: LOOP UNTIL BOTH FOUND
##################################################
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--window", default="Remote Control", help="Which window to capture.")
    args=parser.parse_args()

    ensure_dirs()

    while True:
        # 1) Check class 0
        img0 = capture_window(args.window)
        found0 = detect_class(img0, class_id=0, conf_thres=CONF_THRESHOLD)
        if found0:
            print("[INFO] Class 0 (PlateUp!) found => no labeling needed.")
        else:
            print("[INFO] Class 0 not found => user labeling.")
            user_label_image(img0, class_id=0)
            print("[INFO] Running fullTrain.py for new data.")
            subprocess.run(["python","fullTrain.py"])
            continue  # after training, repeat the loop

        # 2) Wait 3 seconds
        print("[INFO] Wait 3 seconds, then check class 1.")
        time.sleep(3)

        # 3) Check class 1
        img1 = capture_window(args.window)
        found1 = detect_class(img1, class_id=1, conf_thres=CONF_THRESHOLD)
        if found1:
            print("[INFO] Class 1 (Factorio) found => no labeling needed.")
        else:
            print("[INFO] Class 1 not found => user labeling.")
            user_label_image(img1, class_id=1)
            print("[INFO] Running fullTrain.py for new data.")
            subprocess.run(["python","fullTrain.py"])
            continue  # after training, repeat the loop

        # if we reach here => both found in one pass
        print("[INFO] Both classes found in this iteration => done!")
        break

    print("[INFO] Exiting script. We have both classes now.")


if __name__=="__main__":
    main()
