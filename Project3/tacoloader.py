import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image, ExifTags
from pycocotools.coco import COCO

class TacoDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, size, train=False, validation=False, num_classes=28):
        self.dataset_path = data_path
        self.anns_file_path = self.dataset_path + '/' + 'annotations.json'
        self.transform = transform
        self.size = size
        self.num_classes = num_classes
        # Read annotations
        with open(self.anns_file_path, 'r') as f:
            dataset = json.loads(f.read())
        categories_df = pd.DataFrame(dataset['categories'])
        categories_df['label'] = categories_df['supercategory'].rank(method='dense', ascending=True).astype(int)
        self.categories_dict = dict(zip(categories_df['id'], categories_df['label']))
        
        # Loads dataset as a coco object
        self.coco = COCO(self.anns_file_path)

        # Get image ids
        imgIds = []

        for category_name in categories_df["supercategory"]:
            catIds = self.coco.getCatIds(supNms=[category_name])
            for catId in catIds:
                imgIds += (self.coco.getImgIds(catIds=catId))
            imgIds = list(set(imgIds))

        np.random.seed(2)  
        np.random.shuffle(imgIds)
        n = len(imgIds)
        imgIds_train, imgIds_validation, imgIds_test = imgIds[:int(0.70*n)], imgIds[int(0.70*n):int(0.85*n)], imgIds[int(0.85*n):]
        
        if train is True:
            self.imgIds = imgIds_train
        elif validation is True:
            self.imgIds = imgIds_validation
        else:
            self.imgIds = imgIds_test

    def __len__(self):
        'Returns the total number of samples'
        return len(self.imgIds)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        img =  self.coco.loadImgs(self.imgIds[idx])[0]
        image_path = self.dataset_path + '/' +img['file_name']
        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        I = Image.open(image_path)
        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)
        X = self.transform(I)
        
        # Load annotations
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
        anns_sel = self.coco.loadAnns(annIds)
        bboxes = []
        labels_idx = []
        
        for ann in anns_sel:
            kx = self.size / img["width"]
            ky = self.size / img["height"]
            bboxes.append(np.array([kx, ky, kx, ky]) * ann['bbox'])
            labels_idx.append(self.categories_dict[ann["category_id"]]-1)
            
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.zeros(self.num_classes)
        labels[labels_idx] = 1
        return X, bboxes, labels
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels
    