"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import numpy as np  # for matrix maths
import torch
import torch.nn as nn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader import RotNetDataset, RotationAndContrastiveDataset
from logger import logger
from losses import ContrastiveLoss
from models import Generator, RotnetDiscriminator, ContrastiveDiscriminator
from utils import (DEVICE, create_rot_transforms, plot_sample_images,
                   weights_init, plot_gan_loss_plots, create_con_transforms)
import pickle
# from ray import tune
from fid_score.fid_score import FidScore
from itertools import product
import os
from pytorchtools import EarlyStopping

def train(config):

    torch.manual_seed(999)

    # create a dataset
    images_path = "/Users/oleksandrmakarevych/Documents/AWS_Keys/Code/SSLGAN/Images_Test"
    # dataset = RotNetDataset(images_path,
    #                         transform=create_rot_transforms(),
    #                         use_rotations=True)

    dataset = RotationAndContrastiveDataset(
        images_path,
        given_transforms= [
            create_rot_transforms(),
            create_con_transforms()
        ],
        use_rotations=True
    )

    netD_rotnet = RotnetDiscriminator().to(DEVICE())
    netD_rotnet.apply(weights_init)

    netD_contr = ContrastiveDiscriminator().to(DEVICE())
    netD_contr.apply(weights_init)

    netG = Generator().to(DEVICE())
    netG.apply(weights_init)

    img_list = []
    G_losses = []
    D_losses = []
    plot_samples = "/home/ubuntu/udl/plots/sample/rotnet_contr/image.png"
    iters = 0
    num_epochs = config['epochs']
    real_label = 1
    fake_label = 0
    device = DEVICE()
    criterion = nn.BCELoss()
    self_inducing_criterion = nn.CrossEntropyLoss()
    self_inducing_criterion_contr = ContrastiveLoss(64, temperature=0.5)

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    optimizerD = Adam(netD_rotnet.parameters(), lr=config['lr1'])
    optimizerG = Adam(netG.parameters(), lr=config['lr2'])
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    print('Using device', device)
    # Commented out IPython magic to ensure Python compatibility.
    print("Starting Training Loop...")
    # For each epoch
    # early_stopping = EarlyStopping(patience=5, verbose = True)
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):

            # Train with real data dist
            netD_rotnet.zero_grad()
            # TODO: Stat HERE
            # Format batch
            real_cpu = data[0].to(device)
            target_label = data[1].to(device)
            b_size = real_cpu.size(0)
            # print(b_size)
            label = torch.full((b_size, ),
                               real_label,
                               dtype=torch.float,
                               device=device)
            # Forward pass real batch through D
            output, angle_pred = netD_rotnet(real_cpu, self_learning=True)
            # Calculate loss on real data dist
            errD_real = criterion(output.view(-1), label)
            # compute rotation error
            errD_rotation = self_inducing_criterion(angle_pred, target_label)

            #Contr Part
            real_cpu = data[2].to(device)
            augmented_image = data[3].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,),
                               real_label,
                               dtype=torch.float,
                               device=device)
            # forward pass for the real samples over discriminator.
            output, projection = netD_contr(real_cpu, self_learning=True)
            aug_projection = netD_contr.forward(augmented_image,
                                          self_learning=True,
                                          discriminator=False)
            # Calculate loss on all-real batch
            errD_real_contr = criterion(output.view(-1), label)
            errD_ssl_contr = self_inducing_criterion_contr(projection,
                                               aug_projection)  # contrastive loss


            # Calculate gradients for D in backward pass
            errD_real_rot = errD_real + errD_rotation + errD_real_contr + errD_ssl_contr
            errD_real_rot.backward()
            D_x = output.mean().item()

            ## Train the discriminator with the fake data dist
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD_rotnet(fake.detach()).view(-1)
            output_contr = netD_contr(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label) + criterion(output_contr, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real_rot + errD_fake
            # Update D
            optimizerD.step()

            # network updater for the generator
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD_rotnet(fake).view(-1)
            output_contr = netD_contr(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label) + criterion(output_contr, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 2 == 0:
                logger.info(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(),
                       errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            #TODO: End Here

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 2 == 0) or ((epoch == num_epochs - 1) and
            #                           (i == len(dataloader) - 1)):
                # plot_name = ""
                # with torch.no_grad():
                #     fake = netG(fixed_noise).detach().cpu()
                #     output = vutils.make_grid(fake, padding=2, normalize=True)
                #     _path_to_plot = plot_samples.split(".")
                #     path_to_samples = f"{_path_to_plot[0]}_{epoch}_{iters}.{_path_to_plot[-1]}"
                #     plot_sample_images(output, path_to_samples)

            iters += 1

        # os.system('rm -rf plots/')
        # for i in range(1000):
        #     with torch.no_grad():
        #         noise = torch.randn(1, 100, 1, 1, device=DEVICE())
        #         fake = netG(noise).detach().cpu()
        #         output = vutils.make_grid(fake, padding=2, normalize=True)
        #         _path_to_plot = plot_samples.split(".")
        #         path_to_samples = f"{_path_to_plot[0]}_{i}.{_path_to_plot[-1]}"
        #         plot_sample_images(output, path_to_samples)
            # noise = torch.randn(1, 100, 1, 1, device=DEVICE())
            # model = torch.load(args.model_path, map_location=DEVICE())
            # output = model(noise)
            # image_grid = vutils.make_grid(output, padding=100, normalize=True)
            # plot_sample_images(image_grid.cpu(), args.image_path+str(i), show_image=args.show_image)

        # fid = FidScore(
        #     paths=[
        #         '/home/ubuntu/udl/Images_Test',
        #         '/home/ubuntu/udl/plots/sample/rotnet'
        #     ],
        #     device=device,
        #     batch_size=32
        # )
        # fid = fid.calculate_fid_score()
        # early_stopping(fid, netG)
        # print('FID: ', fid, ' for epoch: ', epoch)
        # if early_stopping.early_stop:
        #     print('Stopping after epochs', epoch)
        #     print('FID: ', fid)
        #     with open('result.txt', 'a') as f:
        #         print('config: ', config, ' FID: ', fid, file=f)
        #     break


    # os.system('rm -rf plots/')
    # for i in range(1000):
    #     with torch.no_grad():
    #         noise = torch.randn(1, 100, 1, 1, device=DEVICE())
    #         fake = netG(noise).detach().cpu()
    #         output = vutils.make_grid(fake, padding=2, normalize=True)
    #         _path_to_plot = plot_samples.split(".")
    #         path_to_samples = f"{_path_to_plot[0]}_{i}.{_path_to_plot[-1]}"
    #         plot_sample_images(output, path_to_samples)
        # noise = torch.randn(1, 100, 1, 1, device=DEVICE())
        # model = torch.load(args.model_path, map_location=DEVICE())
        # output = model(noise)
        # image_grid = vutils.make_grid(output, padding=100, normalize=True)
        # plot_sample_images(image_grid.cpu(), args.image_path+str(i), show_image=args.show_image)

    # fid = FidScore(
    #     paths=[
    #         '/home/ubuntu/udl/Images_Test',
    #         '/home/ubuntu/udl/plots/sample/rotnet'
    #     ],
    #     device=device,
    #     batch_size=32
    # )
    # fid = fid.calculate_fid_score()
    # print('FID: ', fid)
    # with open('result.txt', 'a') as f:
    #     print('config: ', config, ' FID: ', fid, file=f)


    # noise = torch.randn(1, 100, 1, 1, device=DEVICE())

    # plot_gan_loss_plots(D_losses,G_losses, "loss")
    # torch.save(netG, "generator_rotnet.pt")
    # tune.report(fid = fid)

    # with open('losses.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([D_losses, G_losses ], f)


if __name__ == '__main__':
    d = {
        'epochs': [150, 200, 250],
        'batch_size': [64, 128, 256, 512],
        'lr1': [1e-3, 1e-4, 1e-5],
        'lr2': [1e-3, 1e-4, 1e-5]
    }
    d_test = {
        'epochs': [2],
        'batch_size': [64],
        'lr1': [1e-3],
        'lr2': [1e-3]
    }
    configs = [dict(zip(d, v)) for v in product(*d_test.values())]
    for config in configs:
        # os.system('rm -rf plots/')
        print(config)
        train(config)


