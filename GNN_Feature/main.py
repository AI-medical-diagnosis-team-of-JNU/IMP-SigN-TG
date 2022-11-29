import torch
import logging
import os
import argparse

from torch import optim
import torch.nn as nn

from model import GAT_Unet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy
import scipy.misc
import sys

from dataset import BasicDataset

dir_img= '../crop/mask/'

dir_checkpoint = './checkpoints/'

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def train_net(
        net,
        device,
        width,
        height,
        epochs = 5,
        batch_size = 16,
        lr = 0.001,
        save_cp = True):
    dataset = BasicDataset(dir_img)

    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    basic_channel=4

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)

    criterion = nn.BCELoss()

    adj = torch.rand((batch_size,batch_size),requires_grad=True).to(device=device)

    # adj = torch.FloatTensor((batch_size, batch_size), requires_grad=True).to(device=device)
    print("adj:",adj)
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for batch in train_loader:

            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_true = image

            # print("mask:", type(mask_true.numpy()))
            # print(mask_true.numpy().shape)
            # print("mask_0:", np.sum(mask_true.numpy()==255.0),np.sum(mask_true.numpy()==0))
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            mask_true = mask_true.to(device=device, dtype=torch.float32)


            mask_pred, _ = net(image, adj)

            loss = criterion(mask_pred, mask_true)

            epoch_loss += loss.item()

            print("epoch[", epoch, "] "," Loss: ", loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            if global_step%1000 == 0:
                try:
                    os.mkdir('./output')
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                for idx in range(batch_size):
                    save_images(mask_pred[idx].detach().cpu().numpy().reshape(1, height, width, 1), image_manifold_size(1),
                        './output/train-'+str(epoch)+'-'+str(idx)+'.png')

            global_step += 1

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if epoch % 200 ==0 :
                torch.save(net.state_dict(),dir_checkpoint+f'model.path')
                np.save("adj.npy",adj.detach().cpu().numpy())
                logging.info(f'Checkpoint {epoch+1} saved')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the GRNN on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=7,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-ht', '--height', metavar='H', type=int, default=32,
                        help='height', dest='height')
    parser.add_argument('-w', '--width', metavar='W', type=int, default=64,
                        help='width', dest='width')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-d', '--dropout', metavar='DR', type=float, nargs='?', default=0.6,
                        help='dropout', dest='dr')
    parser.add_argument('-a', '--alpha', metavar='ALPHA', type=float, nargs='?', default=0.2,
                        help='Leaky-elu', dest='alpha')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    torch.cuda.set_device(1)
    net = GAT_Unet(in_channel=1, dropout=args.dr, alpha=args.alpha, height=args.height, width=args.width, batch_size=args.batchsize, basic_channel = 4)
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)
    net.cuda()
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  height=args.height,
                  width=args.width,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

