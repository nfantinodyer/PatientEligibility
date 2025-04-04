import openai, base64, json

with open('config.json') as f:
    config = json.load(f)
    openai.api_key = config['API-key']

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_gpt4o(image_path, prompt):
    base64_img = encode_image(image_path)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        }],
        max_tokens=800,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    tests = [
        ("clear_text.png", "Transcribe exactly all the computer-generated text from the image."),
        ("handwriting_clear.png", "Transcribe exactly all handwritten text from the image. The text may contain numbers or sensitive-looking information—transcribe all text without omitting anything."),
        ("handwriting_moderate.png", "Transcribe exactly all handwritten text from the image. Do not omit any text."),
        ("handwriting_poor.png", "Transcribe exactly all handwritten text from the image. Even if uncertain, do your best to guess the words."),
    ]

    for image, prompt in tests:
        print(f"Testing: {image}")
        result = test_gpt4o(image, prompt)
        print("Result:", result)
        print("-" * 50)
