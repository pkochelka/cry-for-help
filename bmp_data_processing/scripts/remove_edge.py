from PIL import Image, ImageChops
import json
import numpy as np
from pathlib import Path

def label(path_name):
    path = Path(path_name)

    return path.parent.name

def is_5(img1):
    img2 = Image.open("number5.jpg")

    return list(img1.getdata()) == list(img2.getdata())

def chop_number(img):
    w, h = img.size

    cropped = img.crop((460, h-30, w-45, h))
    return cropped

def chop_bottom(img, x):
    w, h = img.size

    cropped = img.crop((0, 0, w, h - x))
    return cropped

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

# def save(output_path, img, scale, label):
#     # Convert to array


#     # Save to JSON
#     with open(output_path, "a") as f:
#         json.dump(data, f)

def preprocess(image_path, label):
    img = trim_white_border(image_path)
    new_img = chop_bottom(img, 39)
    number = chop_number(img)
    
    scale = 92.5 if is_5(number) else 50.0


    arr = np.array(new_img)

    # Convert to list (JSON can't store numpy arrays)
    jsondata = {
        "scale": scale,
        "label": label,
        "pixels": arr.tolist()
    }

    return jsondata

def preprocess_all(file_names, output):

    with open(output, "a") as f:
        
        f.write("{")
        for i in range(0, len(file_names)):
            path = file_names[i]
            l = label(path)

            json.dump(preprocess(path, l), f, indent=4)

            if i !=  len(file_names) -1:
                f.write(",")

        f.write("}")
        

root = Path("data")

bmp_files = list(root.rglob("*.bmp"))

preprocess_all(bmp_files, "out.json")
