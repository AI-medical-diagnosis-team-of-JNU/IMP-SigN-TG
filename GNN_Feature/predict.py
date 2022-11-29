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

num = 2

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
    for i in range(num+1):

        adj = np.load("./adj.npy")
        adj = torch.from_numpy(adj).to(device=device)

        dir = '../crop/T0/' + str(i) + '/'
        dataset = BasicDataset(dir)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        for batch in train_loader:

            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_pred, feature = net(image, adj)
            np.save('../crop/T0/f_'+str(i)+'.npy', feature.detach().cpu())

        dir = '../crop/T1/' + str(i) + '/'
        dataset = BasicDataset(dir)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        for batch in train_loader:

            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_pred, feature = net(image, adj)
            np.save('../crop/T1/f_'+str(i)+'.npy', feature.detach().cpu())

        dir = '../crop/T2/' + str(i) + '/'
        dataset = BasicDataset(dir)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        for batch in train_loader:

            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_pred, feature = net(image, adj)
            np.save('../crop/T2/f_'+str(i)+'.npy', feature.detach().cpu())


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

