import os
import shutil

TINY_IMAGENET_ROOT = './data/tiny-imagenet-200/'
VAL_DIR = os.path.join(TINY_IMAGENET_ROOT, 'val')
ANNOTATIONS_FILE = os.path.join(VAL_DIR, 'val_annotations.txt')
IMAGES_DIR = os.path.join(VAL_DIR, 'images')
    
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = f.readlines()

for line in annotations:
    parts = line.strip().split('\t')
    img_filename = parts[0]
    class_name = parts[1]
    
    class_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    src_path = os.path.join(IMAGES_DIR, img_filename)
    dest_path = os.path.join(class_dir, img_filename)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        
if os.path.exists(IMAGES_DIR) and not os.listdir(IMAGES_DIR):
    os.rmdir(IMAGES_DIR)
    os.remove(ANNOTATIONS_FILE)
    print("Success")
elif os.path.exists(IMAGES_DIR) and os.listdir(IMAGES_DIR):
    print("Error")
else:
    print("Success (some issues)")