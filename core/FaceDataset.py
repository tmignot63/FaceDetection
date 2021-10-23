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
        """Constructor of the FaceDataset

        Args:
            root (string): root path 
            transform (transforms, optional): transforms to apply on pictures. Defaults to None.
            face (bool, optional): True if it's only faces, False if it's only non faces. Defaults to True.
            scenery (bool, optional): True if it's the boostrapping dataset. Defaults to False.
            test (bool, optional): True if it's a test dataset. Defaults to False.
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
        """Get an item in the dataset

        Args:
            idx (int): id of the item

        Returns:
            Tensor,int,string: returns the image,label and image's name
        """
        image_name = self.dataset[idx]

        if(self.face == True):
            label = 1
        else:
            label = 0

        imageTransform = Image.open(
            self.root+"/"+str(image_name)).convert("RGB")

        if self.transform != None:
            imageTransform = self.transform(imageTransform)

        return (imageTransform, label, image_name)

    def create_dataset(self, path):
        """Creates the dataset

        Args:
            path (string):root path
        """
        dataset = self.dataset

        if(self.test == True):
            # testing case
            for files in os.listdir(path):
                if os.path.isdir(os.path.join(path, files)) == False:
                    dataset.append(files)
            return

        if(self.face == True):
            # if faces
            for files in os.listdir(path+"/1"):
                dataset.append("/1/"+files)

        else:
            if self.scenery == True:
                # boostrapping case
                for files in os.listdir(path+"/scenery/36_36"):
                    dataset.append("/scenery/36_36/"+files)

            else:
                for categories in os.listdir(path+"/0"):
                    for files in os.listdir(path+"/0/"+categories):
                        dataset.append("/0/"+categories+"/"+files)
