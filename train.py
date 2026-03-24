from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform   
from models.VAE_M_S_R_Net import  Discriminator, VAE_MSRNet
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import argparse
import time
import torch
import os

import math
import cv2

import torch.nn.functional as F

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def get_parser():
    parser = argparse.ArgumentParser(
        prog='VAE_M-S-R Net',
        usage='python3 main.py',
        description='This module demonstrates shadow removal using VAE_M-S-R Net.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.993, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=2e-4)

    return parser

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(G1, dataset, device, filename):
    img, gt_shadow, gt = zip(*[dataset[i] for i in range(9)])
    img = torch.stack(img)
    gt_shadow = torch.stack(gt_shadow)
    gt = torch.stack(gt)
    print(gt.shape)
    print(img.shape)

    with torch.no_grad():
        #reconstruct_input, reconstruct_tf, reconstruct_gt = G1(img.to(device), gt.to(device))
        img_mask = torch.cat([img, gt_shadow], dim=1)  # Added mask channel
        reconstruct_tf, tg1, td1 = G1.test_pair(img_mask.to(device))
        
        # reconstruct_tf, tg1, td1 = G1.test_pair(img.to(device))
        
        grid_rec = make_grid(unnormalize(reconstruct_tf.to(torch.device('cpu'))), nrow=3)
        print(grid_rec.shape)
        tg1 = tg1.to(torch.device('cpu'))
        td1 = td1.to(torch.device('cpu'))
        reconstruct_tf = reconstruct_tf.to(torch.device('cpu'))

    grid_removal = make_grid(torch.cat((unnormalize(img), unnormalize(gt),
                                        unnormalize(tg1),
                                        unnormalize(td1),
                                        unnormalize(reconstruct_tf)), dim=0), nrow=9)

    save_image(grid_rec, filename + 'shadow_removal_img.jpg')
    save_image(grid_removal, filename+'_removal_separation.jpg')

def plot_log(data, save_model_name='model'):
    plt.cla()

    # Plot all losses except VAE_loss
    plt.plot(data['G'], label='G_loss')
    plt.plot(data['D'], label='D_loss')
    plt.plot(data['SG'], label='Single_Generator_loss')
    plt.plot(data['GENERAL'], label='General_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')
    plt.close()

    # Plot VAE_loss in a separate graph
    plt.cla()
    plt.plot(data['VAE'], label='VAE_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('VAE Loss')
    plt.savefig('./logs/'+save_model_name+'_vae_only.png')
    plt.close()
    
def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')

def train_model(G1, D1, dataloader, val_dataset, num_epochs, parser,  save_model_name='model'):
    check_dir()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    D1.to(device)
    
    """use GPU in parallel"""
    # if device == 'cuda':
    #     G1 = torch.nn.DataParallel(G1)
    #     D1 = torch.nn.DataParallel(D1)
    #     print("parallel mode")

    print("device:{}".format(device))

    lr = parser.lr
    beta1, beta2 = 0.5, 0.999

    optimizerG = torch.optim.Adam([{'params': G1.parameters()}], lr=lr, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, 'min', factor=0.6, verbose=True, threshold=0.00001, min_lr=0.000000000001, patience=30)
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}], lr=lr, betas=(beta1, beta2))
   

    criterionGAN = nn.BCEWithLogitsLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    reconstruction_loss = nn.MSELoss()

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'lambda1': 10, 'lambda2': 0.1, 'lambda3': 0.2}

    g_losses = []
    d_losses = []
    general_losses = []
    single_gan_losses = []
    vae_losses = []

    for epoch in range(num_epochs + 1):
        G1.train()
        D1.train()


        t_epoch_start = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_single_g_loss = 0.0
        epoch_general_loss = 0.
        epoch_vae_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')
        data_len = len(dataloader)
        for n, (images, gt_shadow, gt) in enumerate(tqdm(dataloader)):
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            gt = gt.to(device)
            gt_shadow = gt_shadow.to(device)
            
            images_mask = torch.cat([images, gt_shadow], dim=1)
            black_mask = torch.zeros_like(gt_shadow)
            gt_mask = torch.cat([gt, black_mask], dim=1)

            mini_batch_size = images.size()[0]

            # Train Discriminator
            set_requires_grad([D1], True)  # enable backprop$
            optimizerD.zero_grad()

            # for D1
            reconstruct_input, reconstruct_tf, reconstruct_gt = G1(images_mask, gt_mask)
            #print(general_loss)

            fake1 = torch.cat([images, reconstruct_tf], dim=1)
            real1 = torch.cat([images, gt], dim=1)

            out_D1_fake = D1(fake1.detach())
            out_D1_real = D1(real1)

            label_D1_fake = Variable(Tensor(np.zeros(out_D1_fake.size())), requires_grad=True)
            label_D1_real = Variable(Tensor(np.ones(out_D1_fake.size())), requires_grad=True)

            loss_D1_fake = criterionGAN(out_D1_fake, label_D1_fake)
            loss_D1_real = criterionGAN(out_D1_real, label_D1_real)

            D_loss = lambda_dict["lambda2"] * loss_D1_fake + lambda_dict["lambda2"] * loss_D1_real
            D_loss.backward()
            optimizerD.step()

            epoch_d_loss += D_loss.item()

            set_requires_grad([D1], False)
            optimizerG.zero_grad()

            fake1 = torch.cat([images, reconstruct_tf], dim=1)
            out_D1_fake = D1(fake1.detach())
            G_L_CGAN1 = criterionGAN(out_D1_fake, label_D1_real)

            G_L_data1 = criterionL1(reconstruct_input, images)
            G_L_data2 = criterionL1(reconstruct_tf, gt)
            G_L_data3 = criterionL1(reconstruct_gt, gt)

            G_loss = lambda_dict["lambda1"] * G_L_data1 + lambda_dict["lambda1"] * G_L_data2 + \
                     lambda_dict["lambda1"] * G_L_data3 + lambda_dict["lambda2"] * G_L_CGAN1

            epoch_g_loss += G_loss.item()
            epoch_single_g_loss += G_L_CGAN1.item()

            #1st recon_x loss1
            recon_x_1, mu_1, logvar_1 = G1.domain1_encoder.vae(G1.domain1_encoder.x5)
            recon_loss_1 = reconstruction_loss(recon_x_1, G1.domain1_encoder.x5)
            kl_loss_1 = -0.5 * torch.sum(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp())
            
            #2nd recon_x loss2
            recon_x_2, mu_2, logvar_2 = G1.domain2_encoder.vae(G1.domain2_encoder.x5)
            recon_loss_2 = reconstruction_loss(recon_x_2, G1.domain2_encoder.x5)
            kl_loss_2 = -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
            
            #3rd recon_x loss3
            recon_x_3, mu_3, logvar_3 = G1.general_encoder.vae(G1.general_encoder.x5)
            recon_loss_3 = reconstruction_loss(recon_x_3, G1.general_encoder.x5)
            kl_loss_3 = -0.5 * torch.sum(1 + logvar_3 - mu_3.pow(2) - logvar_3.exp())
            
            vae_loss = recon_loss_1 + kl_loss_1 + recon_loss_2 + kl_loss_2 + recon_loss_3 + kl_loss_3
            epoch_vae_loss += vae_loss.item()
            
            G_loss = G_loss + vae_loss
            G_loss.backward()
            optimizerG.step()

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f} || '
              'Epoch_Single_G_Loss:{:.4f} || Epoch_VAE_Loss:{:.5f} || lr:{:.10f}'.format(epoch,
                                                                           epoch_d_loss / (lambda_dict["lambda2"] * 2 * data_len),
                                                                           epoch_g_loss / data_len,
                                                                           epoch_single_g_loss / data_len,
                                                                           epoch_vae_loss / data_len,
                                                                           optimizerG.param_groups[0]["lr"]))

        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        d_losses += [epoch_d_loss / (lambda_dict["lambda2"] * 2 * data_len)]
        g_losses += [epoch_g_loss / data_len]
        general_losses += [epoch_general_loss / data_len]
        single_gan_losses += [epoch_single_g_loss / data_len]
        vae_losses += [epoch_vae_loss / data_len]
        scheduler.step(epoch_g_loss / data_len)
        t_epoch_start = time.time()

        plot_log({'G': g_losses,
                  'D': d_losses,
                  'SG': single_gan_losses,
                  'GENERAL': general_losses,
                  'VAE': vae_losses}, save_model_name)

        if epoch % 10 == 0:
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_G1_' + str(epoch) + '.pth')
            torch.save(D1.state_dict(), 'checkpoints/' + save_model_name + '_D1_' + str(epoch) + '.pth')

            G1.eval()
            evaluate(G1, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))

    return G1

def main(parser):
    G1 = VAE_MSRNet(input_channels=4, output_channels=3)
    D1 = Discriminator(input_channels=6)

    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        
        G1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VAE_M-S-R-Net_G1_' + parser.load + '.pth')))
        D1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VAE_M-S-R-Net_D1_' + parser.load + '.pth')))

    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)[:20]
    print(len(val_img_list["path_A"]))
    print(val_img_list)
    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list,
                                 img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                 phase='train')
    val_dataset = ImageDataset(img_list=val_img_list,
                               img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                               phase='test_no_crop')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                                   pin_memory=True, prefetch_factor=2)

    G1 = train_model(G1, D1, dataloader=train_dataloader,
                                val_dataset=val_dataset,
                                num_epochs=num_epochs,
                                parser=parser,
                                save_model_name='VAE_M-S-R-Net')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)