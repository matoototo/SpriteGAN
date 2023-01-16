import os
import hashlib
import argparse
import numpy as np
from PIL import Image

def compute_sprites(image_path):
    image = Image.open(image_path)
    image.load()
    image = np.asarray(image, dtype="int32")

    img_x, img_y, *_ = image.shape
    x_stride = y_stride = 32

    x_coordinates = range(0, img_x, x_stride)
    y_coordinates = range(0, img_y, y_stride)

    return (image[x:x+x_stride,y:y+y_stride] for x in x_coordinates for y in y_coordinates)

def check_dir(out_path):
    if not os.path.exists(out_path):
        print(f"{out_path} does not exist. Creating.")
        os.makedirs(out_path)

def save_sprites(out_path, sprites):
    for sprite in sprites:
        h = np.ascontiguousarray(sprite).view(np.uint8)
        md5 = hashlib.sha1(h).hexdigest()

        sprite_img = Image.fromarray(sprite.astype("uint8"))
        sprite_img.save(f"{out_path}/{md5}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Split grid into individual sprites.")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    sprites = compute_sprites(args.image)
    check_dir(args.outdir)
    save_sprites(args.outdir, sprites)
