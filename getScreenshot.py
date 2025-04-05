# self_bootstrap.py
import os
import time
import argparse
import subprocess   # <-- for calling fullTrain.py
import cv2
import numpy as np

import pyautogui
from pywinauto import Desktop
from ultralytics import YOLO

##################################################
# CONFIG
##################################################
DATA_YAML = "data.yaml"  # Path to your train/val sets
BEST_MODEL_DIR = "runs/detect/train/weights"
BEST_PT = os.path.join(BEST_MODEL_DIR, "best.pt")

# Where newly labeled images go
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"

CONF_THRESHOLD = 0.5

##################################################
# CREATE DIRS
##################################################
def ensure_dirs():
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)  # where best.pt goes

##################################################
# CAPTURE WINDOW
##################################################
def capture_window(window_title="Steam"):
    """
    Attempt to find a window by partial title with pywinauto, bring it to front, 
    then do a full-screen screenshot with pyautogui.
    """
    try:
        win = Desktop(backend='uia').window(title_re=f".*{window_title}.*")
        win.set_focus()
        time.sleep(0.5)
        print(f"[INFO] Focused window matching '{window_title}'.")
    except Exception as e:
        print(f"[WARN] Could not focus on window '{window_title}': {e}")

    screenshot = pyautogui.screenshot()
    image_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("[INFO] Captured full-screen screenshot.")
    return image_cv

##################################################
# DETECTION
##################################################
def detect_any_class(image_cv):
    """
    Loads the best model if available (else yolov8s), 
    then tries to detect ANY class above conf=CONF_THRESHOLD.
    Returns the results object from YOLO.
    """
    model_path = BEST_PT if os.path.exists(BEST_PT) else "yolov8s.pt"
    model = YOLO(model_path)
    results_list = model.predict(source=image_cv, imgsz=1920)
    if not results_list:
        return None
    return results_list[0]  # first result

##################################################
# SHOW DETECTIONS & MOUSE
##################################################
def show_detections(image_cv, results, conf_threshold=CONF_THRESHOLD):
    """
    Draw bounding boxes for all classes above conf_threshold.
    Move mouse to highest-conf box center. Then show for 2s.
    """
    if not results or not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return

    disp_img = image_cv.copy()
    max_conf = -1.0
    best_center = None

    for box in results.boxes:
        conf = box.conf[0].item()
        cls_id= int(box.cls[0].item())
        if conf < conf_threshold:
            continue

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().tolist()
        cv2.rectangle(disp_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        if conf > max_conf:
            max_conf = conf
            best_center= (int((x1+x2)//2), int((y1+y2)//2))

    if max_conf >= 0 and best_center:
        cx,cy = best_center
        print(f"[INFO] Highest conf box => conf={max_conf:.3f}, center=({cx},{cy})")
        pyautogui.moveTo(cx, cy, duration=1)
    else:
        print("[INFO] No box found above threshold.")

    wname="Detections"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, 1920, 1080)
    cv2.imshow(wname, disp_img)
    cv2.waitKey(2000)
    cv2.destroyWindow(wname)

##################################################
# USER LABEL (if no detections)
##################################################
drawing=False
sx=sy=ex=ey=-1

def mouse_cb(event, x, y, flags, param):
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

def train_folder_is_empty():
    exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return not any(f.lower().endswith(exts) for f in os.listdir(TRAIN_IMG_DIR))

def user_label_image(image_cv, cls_id=0, use_val=False):
    global sx,sy,ex,ey,drawing
    wname="User Label - draw box (s=save, q=skip)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname,1920,1080)
    cv2.setMouseCallback(wname, mouse_cb)

    sx=sy=ex=ey=-1
    clone=image_cv.copy()

    # if train empty => override use_val
    if train_folder_is_empty():
        use_val=False

    while True:
        temp=clone.copy()
        if sx!=-1 and sy!=-1 and ex!=-1 and ey!=-1:
            cv2.rectangle(temp, (sx,sy), (ex,ey), (0,255,0),2)

        cv2.imshow(wname,temp)
        key=cv2.waitKey(10)&0xFF

        if key==ord('s'):
            if sx==ex or sy==ey:
                print("[WARN] No bounding box drawn.")
                continue

            # pick folder
            img_dir= TRAIN_IMG_DIR if not use_val else VAL_IMG_DIR
            lbl_dir= TRAIN_LABEL_DIR if not use_val else VAL_LABEL_DIR

            tstamp=int(time.time())
            fname=f"user_{tstamp}.jpg"
            img_path=os.path.join(img_dir,fname)
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

            lbl_path=os.path.join(lbl_dir,fname.replace(".jpg",".txt"))
            with open(lbl_path,"w") as lf:
                line=f"{cls_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}"
                lf.write(line+"\n")

            print(f"[INFO] Saved => {img_path}")
            print(f"[INFO] Label => {lbl_path}")

            cv2.destroyWindow(wname)
            return True

        elif key==ord('q'):
            print("[INFO] Skipped labeling.")
            cv2.destroyWindow(wname)
            return False

##################################################
# MAIN
##################################################
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--window", type=str, default="Steam",
                        help="Which window to capture from.")
    args=parser.parse_args()

    ensure_dirs()

    # 1) capture screenshot
    img = capture_window(args.window)

    # 2) try detection
    results = detect_any_class(img)
    if not results or not hasattr(results,'boxes') or len(results.boxes)==0:
        print("[INFO] No boxes => user labeling.")
        user_label_image(img, cls_id=0, use_val=False)
        # after labeling => run fullTrain.py
        print("[INFO] Running fullTrain.py ...")
        subprocess.run(["python", "fullTrain.py"])
        return

    # otherwise => show detection, do nothing
    show_detections(img, results, conf_threshold=CONF_THRESHOLD)
    print("[INFO] Found boxes => no labeling needed. If you want to retrain => run `python fullTrain.py` manually.")


if __name__=="__main__":
    main()
