from utils import getBigRectangles
from crop_images import crop_save_36_36_images_scenery

from utils import getBigRectangles
import sys
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader, dataloader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
from decimal import Decimal
import torch.nn.functional as F
from PIL import Image
from FaceDataset import FaceDataset
from PIL import Image, ImageDraw

IMG_TEST_DIR = "C:/Users/gotam/Desktop/INSA/5IF/YOLO/Visages/WIDER_val/WIDER_val/images/1--Handshaking/"
IMG_36_36_TEST_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/tests_36_36"
NEW_SIZE = (432, 432)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
NUM_WORKERS = 2
PIN_MEMORY = True


def evaluate_opti(net_path):
    """This function highligths all the faces of the images in a directory"""
    crop_save_36_36_images_scenery(IMG_TEST_DIR, image_size=(448, 448), img_scenery_dir=IMG_36_36_TEST_DIR, minBound=Decimal(
        '0.1'), maxBound=Decimal('1.4'), step=10)

    list_datasets = []
    list_data_loaders = []

    transform_scenery = transforms.Compose([transforms.ToTensor(
    ), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    for directories in os.listdir(IMG_36_36_TEST_DIR):
        list_datasets.append(FaceDataset(
            os.path.join(IMG_36_36_TEST_DIR, directories), transform=transform_scenery, face=False, scenery=True, test=True))
        list_data_loaders.append((DataLoader(dataset=list_datasets[-1], batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False), os.path.join(IMG_TEST_DIR, directories)))

    net = torch.load(net_path)
    net = net.to(DEVICE)
    net.eval()

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for data_loaders in list_data_loaders:
            recognisedFacesCenters = []
            recognisedFacesPercentages = []
            recognisedFacesCenterSizes = []

            data_loader = data_loaders[0]
            original_image_path = data_loaders[1]

            im = Image.open(original_image_path)
            im = im.resize((448, 448))
            imageCadres = im.copy()

            for batch_idx, (data, target, img_name) in tqdm(enumerate(data_loader)):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = softmax(net(data))
                for j in range(len(output.data)):
                    """If the frame has a very probability to contain a face then we process"""
                    if(output.data[j][1] > 0.85):

                        img_name_splits = img_name[j].split("___")

                        transiCenter = eval(img_name_splits[1][1:-1])
                        detectedFaceSize = float(
                            img_name_splits[2][0:len(img_name_splits[2])-4])

                        recognisedFacesCenters.append(transiCenter)
                        recognisedFacesCenterSizes.append(detectedFaceSize)
                        recognisedFacesPercentages.append(output.data[j][1])
            """Once all the most plausible frames are gathered, we select only one by face"""
            bigRectangles = getBigRectangles(
                recognisedFacesCenters, recognisedFacesPercentages, recognisedFacesCenterSizes)

            """We draw all the final frames"""
            img1 = ImageDraw.Draw(imageCadres)
            for i in range(len(bigRectangles[0])):
                shape = [bigRectangles[0][i][0]-int(bigRectangles[1][i]/2), bigRectangles[0][i][1]-int(
                    bigRectangles[1][i]/2), bigRectangles[0][i][0]+int(bigRectangles[1][i]/2), bigRectangles[0][i][1]+int(bigRectangles[1][i]/2)]
                img1.rectangle(shape, outline="red")

            imageCadres.show()
            input("Press Enter to continue...")


if __name__ == "__main__":
    evaluate_opti("bestResnetModel3epochs")
