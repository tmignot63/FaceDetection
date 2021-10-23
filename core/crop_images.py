import sys
import os
from tqdm import tqdm
import torch.nn as nn
from decimal import Decimal
from PIL import Image
sys.path.append(os.getcwd())

IMG_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/scenery"
IMG_SCENERY_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/scenery/36_36"
NEW_SIZE = (448, 448)


def crop_save_36_36_images_scenery(img_dir, image_size, img_scenery_dir, minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=10):
    """Crop and save all images before network's predictions
    Args:
        img_dir (string): path to the images to crop
        image_size ((int,int)): image new size before cropping
        img_scenery_dir (string): path where the 36*36 images are going to be saved
        minBound (Decimal, optional): min Scaling Factor for the sliding window. Defaults to Decimal('1.0').
        maxBound (Decimal, optional): max Scaling Factor for the sliding window. Defaults to Decimal('1.0').
        step (int, optional): Step size between windows. Defaults to 10.
    """
    for categories in os.listdir(img_dir):
        if os.path.isdir(os.path.join(img_dir, categories)):
            # case boostraping
            if categories != "36_36":
                for files in tqdm(os.listdir(img_dir+"/"+categories)):
                    image = Image.open(
                        img_dir+str("/"+categories+"/"+files)).convert("RGB")
                    image = image.resize(image_size)
                    # sliding window
                    nb_FP = pyramid(image, files, img_scenery_dir, scale=Decimal(
                        '0.1'), minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=10)

        else:
            # case test
            image = Image.open(img_dir+str("/"+categories)).convert("RGB")
            image = image.resize(image_size)
            os.mkdir(os.path.join(img_scenery_dir, categories))
            # sliding window
            nb_FP = pyramid(image, categories, img_scenery_dir, scale=Decimal(
                '0.1'), minBound=minBound, maxBound=maxBound, step=step, modeTest=True)


def pyramid(image, image_name, img_scenery_dir, scale=Decimal('0.1'), minBound=Decimal('1.0'), maxBound=Decimal('1.0'), step=10, modeTest=False):
    """Sliding windows process 
    Args:
        image (PILImage): image to crop
        image_name (string): image_name
        img_scenery_dir (string): path where the 36*36 images are going to be saved
        scale (Decimal, optional): Scaling step. Defaults to Decimal('0.1').
        minBound (Decimal, optional): min Scaling Factor for the sliding window. Defaults to Decimal('1.0').
        maxBound (Decimal, optional): max Scaling Factor for the sliding window. Defaults to Decimal('1.0').
        step (int, optional): Step size between windows. Defaults to 10.
        count_total (int, optional): total numbers of new images. Defaults to 0.
        modeTest (bool, optional): 2 modes :boostrapping and testing. Defaults to False.

    Returns:
        (int): number of 36*36 images
    """
    w, h = image.size
    transiImage = image
    delta = minBound
    TrainImageSize = 36
    uniqueCount = 0

    while delta <= maxBound:
        # Scaling loop
        sizeTransi = (int(w*delta), int(h*delta))
        resizeImage = image.resize(sizeTransi)
        for height in range(0, (int(h*delta) - TrainImageSize + 1), step):
            for width in range(0, (int(w*delta) - TrainImageSize + 1), step):
                # Sliding window loop
                uniqueCount += 1
                transiImage = resizeImage.crop(
                    (width, height, width+TrainImageSize, height+TrainImageSize))

                transiCenter = (int((width+(TrainImageSize/2))/float(delta)),
                                int((height+(TrainImageSize/2))/float(delta)))
                detectedFaceSize = (h*TrainImageSize)/int(h*delta)

                transiImageCopy = transiImage.copy()
                if(modeTest == True):
                    transiImageCopy.save(
                        img_scenery_dir+"/"+str(image_name)+"/___"+str(transiCenter)+"___"+str(detectedFaceSize)+".jpg")
                else:
                    transiImageCopy.save(
                        img_scenery_dir+"/"+str(image_name)+"___"+str(transiCenter)+"___"+str(detectedFaceSize)+".jpg")

        delta += scale
    return uniqueCount


def main():
    crop_save_36_36_images_scenery(IMG_DIR, NEW_SIZE, IMG_SCENERY_DIR)


if __name__ == "__main__":
    main()
