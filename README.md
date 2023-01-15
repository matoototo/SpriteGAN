# SpriteGAN

<img src="./examples/psi-progression.gif" width=100% alt="PSI progression GIF"></img>

This is a [DiffAugment StyleGAN2](https://github.com/mit-han-lab/data-efficient-gans) trained on 32x32 sprites. The model was trained in early 2022. While the outputs are far from perfect and do exhibit partial mode collapse, to my knowledge this is the SOTA in equipment sprite generation at a fixed size.

The GIF above showcases 27 randomly sampled points in the latent space, with increasing truncation_psi from 0.0 to 1.2. A larger example with 2500 individual sprites (at 1.0 psi) can be found under [./examples/psi-1.0.png](examples/psi-1.0.png). If you want to generate your own, follow the installation instructions below.

## Installation

To install, follow the [DiffAugment StyleGAN2](https://github.com/mit-han-lab/data-efficient-gans) installation instructions, download the weights (see [./weights/README.txt](weights/README.txt)) and move both the weights and grid_gen.py to the DiffAugment pytorch root directory.

## Generation

To generate your own sprites, simply call grid_gen.py with the downloaded weights and an output directory. For example, generating a grid similar to the one found under [./examples/psi-1.0.png](examples/psi-1.0.png) would be:
```bash
python3 grid_gen.py --netpath ./network-snapshot.pkl --outdir ./examples --grid_x 50 --grid_y 50
```

