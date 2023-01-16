# Generation code adapted from original repository.

import datetime
import math
import os
import random
import numpy as np
import PIL.Image
import torch
import legacy
import argparse


def make_transparent(image: PIL.Image, threshold):
    image = image.convert("RGBA")
    datas = image.getdata()

    newData = []
    for item in datas:
        if item[0] < threshold and item[1] < threshold and item[2] < threshold:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    image.putdata(newData)
    return image


def save_image_grid(img, fname, drange, grid_size, threshold):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        image = PIL.Image.fromarray(img[:, :, 0], 'L')
        image = make_transparent(image, threshold)
        image.save(fname)
    if C == 3:
        image = PIL.Image.fromarray(img, 'RGB')
        image = make_transparent(image, threshold)
        image.save(fname)

def load_network(path):
    with open(path, "rb") as f:
        resume_data = legacy.load_network_pkl(f)
        G_ema = resume_data['G_ema'].to(device)
        G = resume_data['G'].to(device)
        return (G_ema, G)

def generate(network, batch_size, grid, truncation_psi = 1.0):
    G_ema, G = network
    grid_x, grid_y = grid
    num_images = grid_x * grid_y

    grid_z = torch.randn([num_images, G.z_dim], device=device).split(batch_size)
    labels = np.array([])
    grid_c = torch.from_numpy(labels).to(device).split(batch_size) * int(math.ceil(num_images / batch_size))
    images = torch.cat(
        [G_ema(z=z, c=c, noise_mode='const', truncation_psi=truncation_psi).cpu() for z, c in zip(grid_z, grid_c)]).numpy()

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=random.randint(0, 1000000))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--psi', type=float, default=1.0)
    parser.add_argument('--grid_x', type=int, default=10)
    parser.add_argument('--grid_y', type=int, default=10)
    parser.add_argument('--threshold', type=int, required=False, default=10, help="Threshold for image transparency.")
    args = parser.parse_args()

    path = args.netpath
    outdir = args.outdir
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = args.batch_size
    psi = args.psi
    grid = (args.grid_x, args.grid_y)

    device = torch.device('cuda')

    os.makedirs(outdir, exist_ok=True)
    net = load_network(path)

    start_time = datetime.datetime.now().microsecond
    images = generate(net, batch_size, grid, psi)

    truncation_psi_str = f"{psi}"
    truncation_psi_str += "0"*(4-len(truncation_psi_str))
    save_image_grid(images, os.path.join(outdir, f'fakes_init-{seed}-psi-{truncation_psi_str}-{start_time}.png'), \
                    drange=[-1, 1], grid_size=grid, threshold=args.threshold)
