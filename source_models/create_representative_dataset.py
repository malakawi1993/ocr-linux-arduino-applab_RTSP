import os
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Create representative dataset")
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--limit', type=int, required=False)

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(dir_path, 'openimages', 'car', 'images')

all_images = []

for f in os.listdir(image_dir):
    if not f.endswith('.jpg'): continue

    img = Image.open(os.path.join(image_dir, f)).convert("RGB")
    img = img.resize((args.width, args.height), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)  # shape: (H, W, 3)

    # --- NORMALIZE TO [-1, 1] ---
    arr = (arr / 127.5) - 1.0

    all_images.append(arr)

    if args.limit is not None and len(all_images) >= args.limit:
        break

all_images_arr = np.stack(all_images, axis=0)
np.save(os.path.join(dir_path, f'repr_dataset_{args.height}_{args.width}.npy'), all_images_arr)
print(f'Written {all_images_arr.shape} to repr_dataset_{args.height}_{args.width}.npy')
