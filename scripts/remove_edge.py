from PIL import Image, ImageChops
import cv2
import pytesseract
import json
import numpy as np

def is_5(img1):
    img2 = Image.open("number5.jpg")

    return list(img1.getdata()) == list(img2.getdata())

def chop_number(image):
    img = Image.open(image)
    w, h = img.size

    cropped = img.crop((460, h-30, w-45, h))
    return cropped

def chop_bottom(image_path, output_path, x):
    img = Image.open(image_path)
    w, h = img.size

    cropped = img.crop((0, 0, w, h - x))
    cropped.save(output_path)

def trim_white_border(image_path):
    img = Image.open(image_path).convert("RGB")

    # Create a white background image
    bg = Image.new(img.mode, img.size, (255, 255, 255))

    # Find difference
    diff = ImageChops.difference(img, bg)

    # Get bounding box of non-white area
    bbox = diff.getbbox()

    if bbox:
        cropped = img.crop(bbox)
        return cropped
    else:
        print("Image is completely white!")

def save(output_path, img, scale, label):
    # Convert to array
    arr = np.array(img)

    # Convert to list (JSON can't store numpy arrays)
    data = arr.tolist()

    data = {
        "scale": scale,
        "label": label,
        "pixels": arr.tolist()
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(data, f)

def preprocess(image_path, output_path):
    img = trim_white_border(image_path)
    data = chop_bottom(img)
    number = chop_number(img, "tmp.png")
    
    scale = 92.5 if is_5(number) else 50.0


    save(output_path, data, scale, "label")

