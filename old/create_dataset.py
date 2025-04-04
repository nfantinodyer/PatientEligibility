import os
import cv2
import time
import json
import argparse
import pyautogui
import numpy as np

"""
A script to generate images and labels for YOLO training.

Workflow:
1) Capture a screenshot using PyAutoGUI.
2) Display it in an OpenCV window.
3) Let the user draw one bounding box with the mouse around the target (e.g., text).
4) Save the screenshot (as .jpg) and a YOLO .txt label file with the bounding-box coordinates.
5) Optionally create a "val" folder (with at least 1 image) to help YOLO validate properly.
6) Optionally create or update a basic 'data.yaml' file to point to train/val folders.

Usage:
  python create_training_materials.py --save_dir data --dataset_type train \
    --count 5 --class_name "PlateUp!"
"""

# ------------------------------------------------
# Global variables used by the mouse callback
# ------------------------------------------------
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y


def reset_box():
    global start_x, start_y, end_x, end_y
    start_x, start_y, end_x, end_y = -1, -1, -1, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data",
        help="Base directory to save images and labels (e.g. 'data')."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        help="Subfolder under images/ and labels/ (e.g. 'train' or 'val')."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="PlateUp!",
        help="Name of the class you want to label (if you have a single class)."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many labeled screenshots to capture."
    )
    parser.add_argument(
        "--create_val",
        action="store_true",
        help="Create at least 1 labeled image in the 'val' folder (if you want YOLO to have a validation set)."
    )
    parser.add_argument(
        "--create_data_yaml",
        action="store_true",
        help="Create a basic 'data.yaml' referencing your train/val. Only does so once or if file doesn't exist."
    )
    args = parser.parse_args()

    # Directory structure
    images_dir = os.path.join(args.save_dir, "images", args.dataset_type)
    labels_dir = os.path.join(args.save_dir, "labels", args.dataset_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Saving images to: {images_dir}")
    print(f"Saving labels to: {labels_dir}")
    print(f"Labeling class: '{args.class_name}'")

    # Create an OpenCV window
    cv2.namedWindow("Labeling")
    cv2.setMouseCallback("Labeling", mouse_callback)

    # We'll capture `args.count` images in the specified dataset_type
    image_index = 0
    while image_index < args.count:
        # 1) Capture screenshot
        screenshot = pyautogui.screenshot()
        # Convert to OpenCV BGR
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        clone = screenshot_cv.copy()

        print(f"\n[INFO] Captured screenshot {image_index+1}/{args.count}.")
        print("[INFO] Draw a bounding box around the target. "
              "Press 's' to save, 'r' to reset, or 'q' to quit.")

        while True:
            display_img = clone.copy()
            if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
                cv2.rectangle(display_img,
                              (start_x, start_y),
                              (end_x, end_y),
                              (0, 255, 0), 2)

            cv2.imshow("Labeling", display_img)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('s'):
                # 's' to save
                if start_x == end_x or start_y == end_y:
                    print("[WARN] No bounding box drawn! Try again.")
                    continue

                # Save the screenshot
                image_name = f"screenshot_{int(time.time())}.jpg"
                image_path = os.path.join(images_dir, image_name)
                cv2.imwrite(image_path, screenshot_cv)
                print(f"[INFO] Saved image: {image_path}")

                # YOLO label file
                # Single-class scenario => class_id=0
                class_id = 0
                img_h, img_w = screenshot_cv.shape[:2]

                # Normalize coords between 0-1
                x1, x2 = sorted([start_x, end_x])
                y1, y2 = sorted([start_y, end_y])
                w = x2 - x1
                h = y2 - y1

                bb_x_center = (x1 + w / 2) / img_w
                bb_y_center = (y1 + h / 2) / img_h
                bb_width = w / img_w
                bb_height = h / img_h

                label_path = os.path.join(labels_dir,
                                          image_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as lf:
                    lf.write(f"{class_id} {bb_x_center:.6f} {bb_y_center:.6f} "
                             f"{bb_width:.6f} {bb_height:.6f}\n")

                print(f"[INFO] Saved label file: {label_path}")
                print(f"[INFO] YOLO coords: class {class_id}, "
                      f"{bb_x_center:.3f}, {bb_y_center:.3f}, "
                      f"{bb_width:.3f}, {bb_height:.3f}")

                reset_box()
                image_index += 1
                break

            elif key == ord('r'):
                print("[INFO] Reset bounding box.")
                reset_box()

            elif key == ord('q'):
                print("[INFO] Quitting early.")
                cv2.destroyAllWindows()
                return

    # Close OpenCV
    cv2.destroyAllWindows()
    print("[INFO] Dataset creation complete in:", images_dir)

    # 2) Optionally create at least 1 labeled image in 'val'
    if args.create_val and args.dataset_type == "train":
        print("[INFO] create_val specified. Creating 1 labeled image in 'val' folder.")
        make_val_image(args.save_dir, args.class_name)

    # 3) Optionally create or update data.yaml
    if args.create_data_yaml:
        generate_data_yaml(args.save_dir)


def make_val_image(base_dir, class_name, class_id=0):
    """
    Creates a single labeled screenshot in data/images/val & data/labels/val.
    This ensures YOLO can do a basic validation pass.
    """
    images_val = os.path.join(base_dir, "images", "val")
    labels_val = os.path.join(base_dir, "labels", "val")
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # Capture screenshot
    screenshot = pyautogui.screenshot()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    print("\n[INFO] Creating a single validation image.")
    print("[INFO] Draw a box and press 's' to save or 'q' to quit. 'r' to reset.")

    clone = screenshot_cv.copy()
    global start_x, start_y, end_x, end_y
    reset_box()

    cv2.namedWindow("val_labeling")
    cv2.setMouseCallback("val_labeling", mouse_callback)

    while True:
        display_val = clone.copy()
        if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
            cv2.rectangle(display_val,
                          (start_x, start_y),
                          (end_x, end_y),
                          (0, 255, 0), 2)

        cv2.imshow("val_labeling", display_val)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            if start_x == end_x or start_y == end_y:
                print("No bounding box drawn! Try again.")
                continue

            image_name = f"val_screenshot_{int(time.time())}.jpg"
            image_path = os.path.join(images_val, image_name)
            cv2.imwrite(image_path, screenshot_cv)
            print(f"[INFO] Saved VAL image: {image_path}")

            img_h, img_w = screenshot_cv.shape[:2]
            x1, x2 = sorted([start_x, end_x])
            y1, y2 = sorted([start_y, end_y])
            w = x2 - x1
            h = y2 - y1

            bb_x_center = (x1 + w / 2) / img_w
            bb_y_center = (y1 + h / 2) / img_h
            bb_width = w / img_w
            bb_height = h / img_h

            label_path = os.path.join(labels_val,
                                      image_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as lf:
                lf.write(f"{class_id} {bb_x_center:.6f} {bb_y_center:.6f} "
                         f"{bb_width:.6f} {bb_height:.6f}\n")
            print(f"[INFO] Saved VAL label file: {label_path}")
            break

        elif key == ord('r'):
            print("[INFO] Reset bounding box.")
            reset_box()
        elif key == ord('q'):
            print("[INFO] Skipping val creation.")
            break

    cv2.destroyWindow("val_labeling")


def generate_data_yaml(base_dir):
    """
    Create a basic data.yaml pointing to images/train & images/val.
    If a data.yaml already exists, we'll just skip creation.
    """
    yaml_path = os.path.join(base_dir, "..", "data.yaml")
    # If you prefer to store data.yaml in the same directory as the images,
    # you can do `yaml_path = os.path.join(base_dir, "data.yaml")` instead.

    if os.path.exists(yaml_path):
        print(f"[INFO] data.yaml already exists at {yaml_path}. Skipping creation.")
        return

    # For single-class use
    yaml_content = f"""train: {os.path.abspath(os.path.join(base_dir, "images", "train"))}
val: {os.path.abspath(os.path.join(base_dir, "images", "val"))}

names:
  0: PlateUp!
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[INFO] Created data.yaml at {yaml_path}:")
    print(yaml_content)


if __name__ == "__main__":
    main()
