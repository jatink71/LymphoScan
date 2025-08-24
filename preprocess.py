import os
import random
from PIL import Image, ImageEnhance
import numpy as np

# Paths to original datasets
DATASETS = {
    "Benign": "C:/Users/jatin/LymphoScan_Project/data/train/Benign",
    "Blood_Cancer": "C:/Users/jatin/LymphoScan_Project/data/train/Blood_Cancer",
    "CLL": "C:/Users/jatin/LymphoScan_Project/data/train/CLL",
    "Early": "C:/Users/jatin/LymphoScan_Project/data/train/Early",
    "FL": "C:/Users/jatin/LymphoScan_Project/data/train/FL",
    "MCL": "C:/Users/jatin/LymphoScan_Project/data/train/MCL",
    "Pre": "C:/Users/jatin/LymphoScan_Project/data/train/Pre",
    "Pro": "C:/Users/jatin/LymphoScan_Project/data/train/Pro"
}

# Define output folders for augmented images
AUGMENTED_DIRS = {key + "_AUG": value + "_AUG" for key, value in DATASETS.items()}

# Number of augmentations per image
AUGMENTATIONS_PER_IMAGE = 5

# Create output folders
for aug_dir in AUGMENTED_DIRS.values():
    os.makedirs(aug_dir, exist_ok=True)

# Apply random augmentations
def augment_image(image):
    if random.random() > 0.5:
        image = image.rotate(random.choice([90, 180, 270]))
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        np_img = np.array(image)
        noise = np.random.normal(0, 10, np_img.shape).astype(np.uint8)
        np_img = np.clip(np_img + noise, 0, 255)
        image = Image.fromarray(np_img)
    return image

# Apply augmentations to each category
for category, input_dir in DATASETS.items():
    output_dir = AUGMENTED_DIRS[category + "_AUG"]
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            path = os.path.join(input_dir, filename)
            image = Image.open(path).convert("RGB")
            image.save(os.path.join(output_dir, filename))
            for i in range(AUGMENTATIONS_PER_IMAGE):
                augmented_image = augment_image(image)
                new_filename = f"{filename.split('.')[0]}_aug_{i}.jpg"
                augmented_image.save(os.path.join(output_dir, new_filename))

print("Data augmentation complete.")