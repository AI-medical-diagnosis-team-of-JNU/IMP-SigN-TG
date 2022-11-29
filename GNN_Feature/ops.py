""" the key parts of the model """

import  torch
from    torch import nn
from    torch.nn import functional as F
import numpy as np
import scipy

# Feature-get module
class Feature_get(nn.Module):

    def __init__(self, in_channels, basic_channel):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, basic_channel, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel, basic_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(basic_channel, basic_channel*2, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel*2, basic_channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel*2),
            nn.ReLU(),
        )
        self.block_3 = nn.Sequential(
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(basic_channel*2, basic_channel*4, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel*4, basic_channel*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel*4),
            nn.ReLU(),
        )
        self.block_4 = nn.Sequential(
            nn.MaxPool2d(2),  # /8
            nn.Conv2d(basic_channel*4, basic_channel*8, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel*8, basic_channel*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel*8),
            nn.ReLU(),
        )
        self.block_5 = nn.Sequential(
            nn.MaxPool2d(2),  # /16
            nn.Conv2d(basic_channel*8, basic_channel*16, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel*16, basic_channel*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel*16),
            nn.ReLU(),
        )
        self.block_6 = nn.Sequential(
            nn.MaxPool2d(2),  # /32
            nn.Conv2d(basic_channel*16, basic_channel*32, kernel_size=3, padding=1),
            nn.Conv2d(basic_channel*32, basic_channel*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(basic_channel*16),
            nn.ReLU(),
        )

    def forward(self, x):
        f_1 = self.block_1(x)
        f_2 = self.block_2(f_1)
        f_3 = self.block_3(f_2)
        f_4 = self.block_4(f_3)
        f_5 = self.block_5(f_4)
        f_6 = self.block_6(f_5)
        return f_6, f_5, f_4, f_3, f_2, f_1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        '''
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Construct adj

def Construct_adj(basic_channel):
    return torch.rand((basic_channel*256,basic_channel*256),requires_grad=True)

# sequential Attention

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))+1e-8
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, feature_o , feature_a):
        n = len(feature_o.size())
        if n == 2:
            batch_size, channel = feature_o.size()
            height = width = 1
        else:
            batch_size, channel, height, width = feature_o.size()

        feat_o = feature_o.view(batch_size, -1, height*width).permute(0, 2, 1).float()
        feat_a = feature_a.view(batch_size, -1, height*width).float()
         
        attention_s = self.softmax(torch.bmm(feat_o, feat_a)).float()
        feat_d = feature_o.view(batch_size, -1, height*width).float()
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1  )).view(batch_size, -1, height, width)
        # feat_e = feat_e.double().cuda()
        # self.alpha = self.alpha.double().cuda()
        # feature_o = feature_o.double().cuda()
        feat_e = feat_e.cuda()
        self.alpha = self.alpha.cuda()
        out = self.alpha * feat_e + feature_o

        return out

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

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
# GCN module

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))+1e-10
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        x = x.double().cuda()
        self.weight = self.weight.double().cuda()

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        xw = xw.double()
        print("support: ",support.dtype, "xw: ",xw.dtype)
        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res
