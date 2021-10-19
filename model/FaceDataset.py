import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

import os
import sys
sys.path.append(os.getcwd())


class FaceDataset(Dataset):
    def __init__(self, root, transform=None, face=True, scenery=False, test=False):
        """[summary]

        Args:
            root ([type]): [description]
            machines ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
            open (bool, optional): [description]. Defaults to True.
        """
        self.root = root
        self.transform = transform
        self.face = face
        self.scenery = scenery
        self.dataset = []
        self.test = test
        self.create_dataset(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_name = self.dataset[idx]

        if(self.face == True):
            label = 1
        else:
            label = 0

        # C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/tests_36_36\\1_Handshaking_Handshaking_1_158.jpg___(106, 106)___30.033519553072626.jpg'

        imageTransform = Image.open(
            os.path.join(self.root, str(image_name))).convert("RGB")

        if self.transform != None:
            imageTransform = self.transform(imageTransform)

        return (imageTransform, label, image_name)

    def create_dataset(self, path):
        dataset = self.dataset

        if(self.test == True):
            for files in os.listdir(path):
                if os.path.isdir(os.path.join(path, files)) == False:
                    dataset.append(files)
            return

        if(self.face == True):
            for files in os.listdir(path+"/1"):
                dataset.append("/1/"+files)

        else:
            if self.scenery == True:
                for files in os.listdir(path+"/scenery/36_36"):
                    dataset.append("/scenery/36_36/"+files)

            else:
                for categories in os.listdir(path+"/0"):
                    for files in os.listdir(path+"/0/"+categories):
                        dataset.append("/0/"+categories+"/"+files)
