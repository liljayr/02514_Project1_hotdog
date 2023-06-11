import torch
import os
import numpy as np
import glob
from PIL import Image, ImageSequence
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

class PH2DataLoader(torch.utils.data.Dataset):
    def __init__(self, size, data_path, train=False, validation=False, augmentation=False):
        'Initialization'
        self.data_paths = sorted(glob.glob(data_path + '/*'))
        self.size = size
        self.augmentation=augmentation

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
    
    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(self.size, self.size))
        color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = resize(image)
        mask = resize(mask)

        if self.augmentation:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Random Color
            if random.random()>0.5:
                image = color(image)
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)

        X, Y =  self.transform(image, label)
        return X, Y
    

class EyeDataLoader(torch.utils.data.Dataset):        
    def __init__(self, transform, data_path, train=False):
        self.transform = transform
        
        image_paths = []
        label_paths = []
        
        image_paths += sorted(glob.glob(data_path + "/images/*"))
        label_paths += sorted(glob.glob(data_path + "/1st_manual/*"))
            
            
        # random seed
        random_seed = 42

        # split dataset into training set and validation set
        train_size = 0.8  # 80:20 split

        images_train, images_valid = train_test_split(image_paths, 
                                                      random_state=random_seed,
                                                      train_size=train_size,
                                                      shuffle=True)

        labels_train, labels_valid = train_test_split(label_paths,
                                                        random_state=random_seed,
                                                        train_size=train_size,
                                                        shuffle=True)
        
        if train:
            self.image_paths = images_train
            self.label_paths = labels_train
        else:
            self.image_paths = images_valid
            self.label_paths = labels_valid
        

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label_gif = Image.open(label_path)
        for frame in ImageSequence.Iterator(label_gif):
            label=frame       

        Y = self.transform(label)
        X = self.transform(image)
        return X, Y