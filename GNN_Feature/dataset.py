from torch.utils.data import Dataset
from os.path import splitext
from os import listdir
import torch
import logging
from glob import glob
from PIL import Image
import numpy as np


class BasicDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        pil_img = np.array(pil_img)
        if len(pil_img.shape) == 2:
            pil_img = np.expand_dims(pil_img, axis=2)

        # HWC to CHW
        img_trans = pil_img.transpose((2, 0, 1))
        img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, item):
        idx = self.ids[item]
        img_file = glob(self.img_dir + idx + ".*")

        img = Image.open(img_file[0])
        img = self.preprocess(img)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
        }
