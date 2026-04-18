import os
import pandas as pd
from PIL import Image
import numpy as np

base_dir = "data"

data = []

for label in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, label)
    
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            if file.lower().endswith(".bmp"):
                file_path = os.path.join(class_dir, file)
                
                img = Image.open(file_path)
                img_array = np.array(img)
                
                data.append({
                    "image": img_array,
                    "label": label
                })

df = pd.DataFrame(data)

print(df.head())