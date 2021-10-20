from utils import getBigRectangles
import sys
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
from decimal import Decimal
import torch.nn.functional as F
from PIL import Image
from FaceDataset import FaceDataset

sys.path.append(os.getcwd())

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 0.00005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0005
EPOCHS = 5
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model1.pth.tar"
IMG_DIR = "C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping"
NB_BOOSTRAP_ITER = 4

transform_train = transforms.Compose([transforms.Resize((36, 36)), transforms.ToTensor(
), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_scenery = transforms.Compose([transforms.ToTensor(
), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def train(train_loader, model, optimizer, loss_fn, epoch, writer):
    """This function trains your system with a given dataset"""
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    model.train()
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
    for batch_idx, (x, y, img_name) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Progress bar
        loop.set_postfix(loss=loss.item(), epoch=epoch)

    print(f"Mean Train loss : {sum(mean_loss)/len(mean_loss)}")
    writer.add_scalar("train loss", sum(mean_loss)/len(mean_loss), epoch)


def validation(model, validation_loader, criterion, epoch, writer, scheduler=None):
    """This function computes the current effeciency of the system"""
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target, img_name) in enumerate(validation_loader):
            count += 1
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= count
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    if(scheduler):
        scheduler.step()
    if writer:
        writer.add_scalar('validation loss', test_loss, epoch)
    return test_loss


def train_val_test(dataset, ratio_val, ratio_test):
    """This function splits datasets for the system's training"""
    training_size = int(len(dataset)*(1-ratio_val-ratio_test))
    val_size = int(len(dataset)*(ratio_val))
    train_idx, val_idx, test_idx = random_split(dataset, [training_size, val_size, len(
        dataset)-val_size-training_size], generator=torch.Generator().manual_seed(42))
    datasets = {}
    datasets['train'] = train_idx
    datasets['val'] = val_idx
    datasets['test'] = test_idx
    return datasets


def boostrap_test(net, loader, epoch, thr_fa, writer, maxFp):
    """This function finds false positives examples and put it in a special folder to improve the system later"""
    net.eval()
    countFP = 0
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_idx, (data, target, img_name) in tqdm(enumerate(loader)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = output = softmax(net(data))
            for j in range(len(output.data)):
                if(output.data[j][1] > thr_fa and output.data[j][1] < 0.9):
                    countFP += 1
                    if(countFP > maxFp):
                        return countFP
                    image = Image.open(IMG_DIR+str(img_name[j]))
                    image.save("C:/Users/gotam/Desktop/INSA/5IF/datasetBoostraping/0/False alarms/" +
                               str(countFP)+"_"+str(epoch)+".jpg")
    if writer != None:
        writer.add_scalar("False alarms", countFP, epoch)
    return countFP


def creates_data_loader():
    """This function creates datasets for the system's training"""
    dataset_faces = FaceDataset(
        IMG_DIR, transform=transform_train, face=True)

    dataset_no_faces = FaceDataset(
        IMG_DIR, transform=transform_train, face=False)

    datasets_faces_split = train_val_test(dataset_faces, 0.2, 0.0)
    datasets_no_faces_split = train_val_test(dataset_no_faces, 0.2, 0.0)

    datasets = {}
    datasets["train"] = datasets_faces_split["train"] + \
        datasets_no_faces_split["train"]
    datasets["test"] = datasets_no_faces_split["test"]
    datasets["val"] = datasets_faces_split["val"] + \
        datasets_no_faces_split["val"]

    train_loader = DataLoader(dataset=datasets["train"], batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    val_loader = DataLoader(dataset=datasets["val"], batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    return train_loader, val_loader


def main():

    writer = SummaryWriter(
        log_dir="logs/VisageClassifier/Boostrapping/Resnet18"+str(EPOCHS) +
        "epochs_lr"+str(LEARNING_RATE))

    """Retrieves an already computed model or computes a new one"""
    if LOAD_MODEL:
        model = torch.load(LOAD_MODEL_FILE).to(DEVICE)
    else:
        model = torchvision.models.resnet18(pretrained=True).to(DEVICE)
        model.fc = torch.nn.Linear(in_features=512, out_features=2)
        model = model.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    dataset_scenery = FaceDataset(
        IMG_DIR, transform=transform_scenery, face=False, scenery=True)

    scenery_loader = DataLoader(dataset=dataset_scenery, batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    train_loader, val_loader = creates_data_loader()

    boost_iter = 0
    thr_fa = 0.3
    lr = LEARNING_RATE

    while boost_iter < NB_BOOSTRAP_ITER:

        best_loss = validation(model, val_loader, loss_fn,
                               0+boost_iter*EPOCHS, writer, scheduler=None)

        for epochs in range(EPOCHS):
            train(train_loader, model, optimizer,
                  loss_fn, (epochs+1)+boost_iter*EPOCHS, writer)
            val_loss = validation(model, val_loader, loss_fn,
                                  epochs+1+boost_iter*EPOCHS, writer, scheduler=None)

            if val_loss < best_loss:
                torch.save(model, "bestResnetModel"+str(EPOCHS) +
                           "epochs_lr"+str(LEARNING_RATE))
                best_loss = val_loss

        boost_iter += 1
        model = torch.load("bestResnetModel"+str(EPOCHS) +
                           "epochs_lr"+str(LEARNING_RATE))
        lr *= 0.5
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        boostrap_test(model, scenery_loader, boost_iter,
                      thr_fa, writer, maxFp=5000)
        if thr_fa >= 0.1:
            thr_fa = thr_fa-0.099

        train_loader, val_loader = creates_data_loader()


if __name__ == "__main__":
    main()
