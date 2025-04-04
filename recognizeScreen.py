import openai
import base64
import json
import re
from pywinauto import Application

# Get API from config.json
with open('config.json') as f:
    config = json.load(f)
    openai.api_key = config['API-key']

def screenshot_steam():
    app = Application().connect(title_re=".*Steam.*")
    window = app.window(title_re=".*Steam.*")
    window.set_focus()
    screenshot_path = "steam_screen.png"
    img = window.capture_as_image()
    img.save(screenshot_path)
    print("Steam screenshot saved.")
    return screenshot_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_screenshot(image_path):
    base64_image = encode_image(image_path)

    prompt_text = (
        "You are an accurate OCR system tasked with extracting text from a Steam application screenshot. "
        "Carefully read each game title exactly as it appears, double-checking for accuracy. "
        "Respond ONLY with a JSON-formatted array of visible game titles, without any additional text or markdown formatting. "
        "Example format: [\"Game 1\", \"Game 2\", \"Game 3\"]"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0.0,  # Ensures deterministic outputs for accuracy
    )

    result = response.choices[0].message.content.strip()
    print("Raw AI response:", result)

    # Remove any potential markdown or extraneous characters
    result_clean = re.sub(r'^```json|```$', '', result).strip()

    # Ensure valid JSON
    try:
        parsed_json = json.loads(result_clean)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Initial parsing failed: {e}")
        # Attempt to fix common issues
        try:
            result_clean = result_clean.replace("\n", "").replace("\\", "")
            parsed_json = json.loads(result_clean)
            return parsed_json
        except json.JSONDecodeError as e2:
            print(f"Fallback parsing failed: {e2}")
            return None

if __name__ == "__main__":
    screenshot_path = screenshot_steam()
    parsed_json = analyze_screenshot(screenshot_path)

    if parsed_json:
        with open("steam_games.json", "w") as json_file:
            json.dump(parsed_json, json_file, indent=4)
        print("Steam games JSON saved.")
    else:
        print("Failed to parse and save JSON.")
