from utils import getBigRectangles
from crop_images import crop_save_36_36_images_scenery

from utils import getBigRectangles
import sys
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from decimal import Decimal
from PIL import Image
from FaceDataset import FaceDataset
from PIL import Image, ImageDraw

# Hyperparameters
IMG_TEST_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/test_val"
IMG_36_36_TEST_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/test_36_36_v2"
MODEL_PATH = "bestResnetModel5epochs_lr8e-05"
NEW_SIZE = (448, 448)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_WORKERS = 2
PIN_MEMORY = True


def evaluate(net_path):
    """
    Evaluate a model on a testing dataset and plot bounding boxes
    Args:
        net_path (string): model's path we want to test 
    """
    crop_save_36_36_images_scenery(IMG_TEST_DIR, image_size=NEW_SIZE, img_scenery_dir=IMG_36_36_TEST_DIR, minBound=Decimal(
        '0.1'), maxBound=Decimal('1.4'), step=10)

    list_datasets = []
    list_data_loaders = []

    transform_scenery = transforms.Compose([transforms.ToTensor(
    ), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Creates one Dataset and one DataLoader per image
    for directories in os.listdir(IMG_36_36_TEST_DIR):
        list_datasets.append(FaceDataset(
            os.path.join(IMG_36_36_TEST_DIR, directories), transform=transform_scenery, face=False, scenery=True, test=True))
        list_data_loaders.append((DataLoader(dataset=list_datasets[-1], batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False), os.path.join(IMG_TEST_DIR, directories)))

    # Load network
    net = torch.load(net_path)
    net = net.to(DEVICE)
    net.eval()

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for data_loaders in list_data_loaders:

            # Initialize lists
            recognisedFacesCenters = []
            recognisedFacesPercentages = []
            recognisedFacesCenterSizes = []

            data_loader = data_loaders[0]
            original_image_path = data_loaders[1]

            im = Image.open(original_image_path)
            # Resize pictures in a lower dimension to reduce computing time
            im = im.resize(NEW_SIZE)
            imageCadres = im.copy()

            for batch_idx, (data, target, img_name) in tqdm(enumerate(data_loader)):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = softmax(net(data))
                for j in range(len(output.data)):
                    # If the frame has a high probability to contain a face then we process
                    if(output.data[j][1] > 0.5):

                        img_name_splits = img_name[j].split("___")

                        transiCenter = eval(img_name_splits[1][1:-1])
                        detectedFaceSize = float(
                            img_name_splits[2][0:len(img_name_splits[2])-4])

                        recognisedFacesCenters.append(transiCenter)
                        recognisedFacesCenterSizes.append(detectedFaceSize)
                        recognisedFacesPercentages.append(output.data[j][1])

            # Once all the most plausible frames are gathered,apply non-max suppression
            bigRectangles = getBigRectangles(
                recognisedFacesCenters, recognisedFacesPercentages, recognisedFacesCenterSizes)

            # We draw all the final frames
            img1 = ImageDraw.Draw(imageCadres)
            for i in range(len(bigRectangles[0])):
                shape = [bigRectangles[0][i][0]-int(bigRectangles[1][i]/2), bigRectangles[0][i][1]-int(
                    bigRectangles[1][i]/2), bigRectangles[0][i][0]+int(bigRectangles[1][i]/2), bigRectangles[0][i][1]+int(bigRectangles[1][i]/2)]
                img1.rectangle(shape, outline="red")

            # Show picture with bboxes
            imageCadres.show()


if __name__ == "__main__":
    evaluate(MODEL_PATH)
