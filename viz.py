"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import argparse  # for parsing options

import torch  # for torch based ops
import torchvision.utils as vutils  # for generation of images

from utils import DEVICE, plot_sample_images

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Script to generate viz samples from the trained generator")
    argparser.add_argument("--model_path",
                           dest="model_path",
                           help="Path to the trained model")
    argparser.add_argument("--image_path",
                           dest="image_path",
                           help="Path to save generated image")
    argparser.add_argument("--show_image",
                           dest="show_image",
                           help="Flag to show generated image",
                           required=False,
                           default=False,
                           type=bool)

    # parse args
    args = argparser.parse_args()
    for i in range(200):
        noise = torch.randn(1, 100, 1, 1, device=DEVICE())
        model = torch.load(args.model_path, map_location=DEVICE())
        output = model(noise)
        image_grid = vutils.make_grid(output, padding=100, normalize=True)
        plot_sample_images(image_grid.cpu(), args.image_path+str(i), show_image=args.show_image)

