import torch
import os
import numpy as np
import glob
import PIL.Image as Image

class PH2DataLoader(torch.utils.data.Dataset):
    def __init__(self, transform, data_path, train=False, validation=False):
        'Initialization'
        self.transform = transform
        self.data_paths = sorted(glob.glob(data_path + '/*'))
        
        if train:
            self.data_paths = self.data_paths[:int(len(self.data_paths)*0.7)]
        elif validation:
            self.data_paths = self.data_paths[int(len(self.data_paths)*0.7): int(len(self.data_paths)*0.85)]
        else:
            self.data_paths = self.data_paths[int(len(self.data_paths)*0.85):]
            
        self.image_paths = []
        self.label_paths = []
        for path in self.data_paths:
            self.image_paths += sorted(glob.glob(path + "/*_Dermoscopic_Image/*.bmp"))
            self.label_paths += sorted(glob.glob(path + "/*_lesion/*.bmp"))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y