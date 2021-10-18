
from utils import getBigRectangles
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from decimal import Decimal
from PIL import Image
sys.path.append(os.getcwd())

IMG_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/scenery"
IMG_SCENERY_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/scenery/36_36"
NEW_SIZE = (432, 432)


def crop_save_36_36_images_scenery(img_dir, image_size, img_scenery_dir, minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=10):
    count_total = 0
    for categories in os.listdir(img_dir):
        if os.path.isdir(os.path.join(img_dir, categories)):
            if categories != "36_36":
                for files in tqdm(os.listdir(img_dir+"/"+categories)):
                    image = Image.open(
                        img_dir+str("/"+categories+"/"+files)).convert("RGB")
                    image = image.resize(image_size)
                    nb_FP = parkour(image, files, img_scenery_dir, scale=Decimal(
                        '0.1'), minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=10,  plotBoxes=False, count_total=count_total)

        else:
            image = Image.open(img_dir+str("/"+categories)).convert("RGB")
            image = image.resize(image_size)
            nb_FP = parkour(image, files, img_scenery_dir, scale=Decimal(
                '0.1'), minBound=minBound, maxBound=maxBound, step=step, plotBoxes=False, count_total=count_total)


def parkour(image, image_name, img_scenery_dir, scale=Decimal('0.1'), minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=5, plotBoxes=False, count_total=0):
    w, h = image.size
    transiImage = image
    imageCadres = image
    delta = minBound
    TrainImageSize = 36
    uniqueCount = 0
    while delta <= maxBound:
        sizeTransi = (int(w*delta), int(h*delta))
        resizeImage = image.resize(sizeTransi)
        for height in range(0, (int(h*delta) - TrainImageSize + 1), step):
            for width in range(0, (int(w*delta) - TrainImageSize + 1), step):
                uniqueCount += 1
                # print(height,width,TrainImageSize)
                transiImage = resizeImage.crop(
                    (width, height, width+TrainImageSize, height+TrainImageSize))

                transiImageCopy = transiImage.copy()
                transiImageCopy.save(

                    img_scenery_dir+"/"+str(image_name)+str(count_total)+str(uniqueCount)+".jpg")
                count_total += 1

        delta += scale
    # if writer != None:
    #     writer.add_scalar("False alarms", countFP, epoch)

    # if plotBoxes:
    #     plot_boxes(recognisedFacesCenters, recognisedFacesPercentages,
    #                recognisedFacesCenterSizes, imageCadres)
    return uniqueCount


def main():
    crop_save_36_36_images_scenery(IMG_DIR, NEW_SIZE, IMG_SCENERY_DIR)


if __name__ == "__main__":
    main()
