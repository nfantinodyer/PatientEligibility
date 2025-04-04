import time
from pywinauto import Application, keyboard
from pywinauto.keyboard import send_keys

# Screenshot Steam window
def screenshot_steam():
    app = Application().connect(title_re=".*Steam.*")
    window = app.window(title_re=".*Steam.*")
    screenshot_path = "steam_screen.png"
    img = window.capture_as_image()
    img.save(screenshot_path)
    print("Steam screenshot saved.")
    return screenshot_path

# Automate Chrome window already opened to ChatGPT
def automate_chrome_chat(prompt, screenshot_path):
    # Connect to Chrome browser window (already opened)
    chrome_app = Application(backend='uia').connect(title_re=".*Chrome.*", timeout=20)
    chrome_window = chrome_app.window(title_re=".*Chrome.*")
    chrome_window.set_focus()
    time.sleep(2)

    # Click on ChatGPT text input
    chrome_window.type_keys("{TAB 10}")  # Adjust TAB count to reliably reach the input box
    time.sleep(1)

    # Upload image by clicking the "+" button via keyboard shortcut (if available)
    # If not, you may manually click or automate coordinates below:
    # send_keys("{VK_TAB}") # Navigate to upload button if necessary
    # send_keys("{ENTER}")
    # time.sleep(1)

    # Open file dialog (Ctrl+O might not work here, so manually)
    print("Please manually click on the '+' button for image upload now.")
    input("After clicking '+', press ENTER here to continue automation...")

    # Automate File Dialog
    file_dialog = Application(backend='uia').connect(title_re=".*Open.*", timeout=20)
    dialog_window = file_dialog.window(title_re=".*Open.*")
    dialog_window.set_focus()
    time.sleep(1)
    
    # Type file path and press Enter
    send_keys(screenshot_path)
    time.sleep(1)
    send_keys("{ENTER}")
    time.sleep(3)  # Wait for image upload completion

    # Type your prompt
    send_keys(prompt, with_spaces=True)
    time.sleep(1)
    send_keys("{ENTER}")

    print("Prompt sent, waiting for response...")
    # Wait for response manually or automate screenshot capture after a fixed wait
    time.sleep(20)  # Adjust according to response speed

    # Optionally, capture the response screenshot
    response_screenshot = chrome_window.capture_as_image()
    response_screenshot.save("chatgpt_response.png")
    print("ChatGPT response screenshot saved.")

if __name__ == "__main__":
    screenshot_path = screenshot_steam()
    prompt_text = "Here's a screenshot of my Steam app. Extract all visible games as a JSON list."
    automate_chrome_chat(prompt_text, screenshot_path)
