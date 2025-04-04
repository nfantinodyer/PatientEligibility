import openai
import base64
import json
import re
import pyautogui
import time
import os
from PIL import Image

COORDS_FILE = 'coordinates.json'

with open('config.json') as f:
    config = json.load(f)
    openai.api_key = config['API-key']

def load_coordinates():
    if os.path.exists(COORDS_FILE) and os.path.getsize(COORDS_FILE) > 0:
        with open(COORDS_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse '{COORDS_FILE}'. Returning empty coords.")
                return {}
    return {}

def save_coordinates(coords):
    with open(COORDS_FILE, 'w') as file:
        json.dump(coords, file, indent=4)

def capture_compressed_screenshot(
    path='full_screen.jpg', max_size=(1920, 1080), quality=30
):
    # Capture with PyAutoGUI
    screenshot = pyautogui.screenshot()
    # Resize
    screenshot = screenshot.resize(max_size)
    # Save as JPEG
    screenshot.save(path, 'JPEG', quality=quality)
    return path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_game_coordinates_from_ai(image_path, game_name):
    base64_image = encode_image(image_path)
    prompt_text = (
        f"Find the exact screen coordinates of the game '{game_name}' in the screenshot. "
        "Respond ONLY in JSON as {\"x\": <value>, \"y\": <value>}."
    )

    # Use gpt-4o or 4o-mini with chat.completions.create
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    prompt_text
                    + "\n[Image attached as base64 - JPEG]\n"
                    + f"data:image/jpeg;base64,{base64_image}"
                ),
            }
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    print("DEBUG: Raw model output:\n", raw)  # <--- Debugging line

    # remove code fences if any
    cleaned = re.sub(r'^```json|```$', '', raw).strip()

    try:
        coords = json.loads(cleaned)
    except json.JSONDecodeError:
        print("ERROR: The model response was not valid JSON!")
        print("Raw response was:", raw)
        raise

    return coords["x"], coords["y"]

def select_game(game_name):
    coords_db = load_coordinates()
    if game_name in coords_db:
        final_x = coords_db[game_name]["x"]
        final_y = coords_db[game_name]["y"]
        print(f"Using stored coordinates for '{game_name}': {final_x},{final_y}")
    else:
        screenshot_path = capture_compressed_screenshot()
        print("Compressed screenshot captured.")

        ai_x, ai_y = get_game_coordinates_from_ai(screenshot_path, game_name)
        print(f"AI approximate coords: ({ai_x}, {ai_y})")

        pyautogui.moveTo(ai_x, ai_y, duration=1)
        user_feedback = input("Is the mouse on the correct spot? (y/n): ")
        if user_feedback.lower() == 'n':
            print("Move your mouse to the correct location and press Enter...")
            input()
            corrected_pos = pyautogui.position()
            final_x, final_y = corrected_pos.x, corrected_pos.y
        else:
            final_x, final_y = ai_x, ai_y

        coords_db[game_name] = {"x": final_x, "y": final_y}
        save_coordinates(coords_db)

    pyautogui.moveTo(final_x, final_y, duration=1)
    pyautogui.click()
    print(f"Clicked on '{game_name}' at: {final_x},{final_y}")

if __name__ == "__main__":
    select_game("PlateUp!")
