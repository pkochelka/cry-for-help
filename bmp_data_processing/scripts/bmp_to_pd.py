from PIL import Image, ImageChops
import numpy as np
import pandas as pd
from pathlib import Path


def label(path_name):
    return Path(path_name).parent.name


def is_5(img1):
    img2 = Image.open("number5.jpg")
    return list(img1.getdata()) == list(img2.getdata())


def chop_number(img):
    w, h = img.size
    return img.crop((460, h - 30, w - 45, h))


def chop_bottom(img, x):
    w, h = img.size
    return img.crop((0, 0, w, h - x))


def trim_white_border(image_path):
    img = Image.open(image_path).convert("RGB")
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    
    print("Image is completely white!")
    return None


def preprocess(image_path, lbl):
    img = trim_white_border(image_path)
    new_img = chop_bottom(img, 39)
    number = chop_number(img)

    scale = 92.5 if is_5(number) else 50.0
    arr = np.array(new_img)

    return {
        "scale": scale,
        "label": lbl,
        "pixels": arr,
        "path": str(image_path),
    }


def preprocess_all(file_names):
    records = []
    n = len(file_names)
    for i, path in enumerate(file_names):
        print(f"{i / n * 100:.1f}%")
        records.append(preprocess(path, label(path)))
    return pd.DataFrame(records)


root = Path("data")
bmp_files = list(root.rglob("*.bmp"))

df = preprocess_all(bmp_files)
print(df.head())

# Optional persistence:
df.to_pickle("out.pkl")