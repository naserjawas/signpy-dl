#
import matplotlib
matplotlib.use("Agg")

#
# from model.lenet import LeNet
from model.rnn import RNN
from model.alexnet import AlexNet
#
from sklearn import metrics
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to ouput trained model")
    ap.add_argument("-p", "--plot", type=str, required=True,
                    help="path to output loss/accuracy plot")
    args  = vars(ap.parse_args())

    #
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10

    #
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT

    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #
    # 1. for LeNet and RNN
    # print("[INFO] loading the KMNIST dataset...")
    # trainData = KMNIST(root="data", train=True, download=True,
    #                 transform=transforms.ToTensor())
    # testData = KMNIST(root="data", train=False, download=True,
    #                 transform=transforms.ToTensor())
    # 2. for AlexNet
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # print("[INFO] loading the KMNIST dataset...")
    # trainData = datasets.KMNIST(root="data", train=True, download=True,
    #                             transform=transform)
    # testData = datasets.KMNIST(root="data", train=False, download=True,
    #                            transform=transform)
    print("[INFO] loading the CIFAR10 dataset...")
    trainData = datasets.CIFAR10(root="data", train=True, download=True,
                                 transform=transform)
    testData = datasets.CIFAR10(root="data", train=False, download=True,
                                transform=transform)

    #
    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
    numValSamples = int(len(trainData) * VAL_SPLIT)
    (trainData, valData) = data.random_split(trainData,
                                        [numTrainSamples, numValSamples],
                                        generator=torch.Generator().manual_seed(42))

    #
    trainDataLoader = data.DataLoader(trainData, shuffle=True,
                                batch_size=BATCH_SIZE)
    valDataLoader = data.DataLoader(valData, batch_size=BATCH_SIZE)
    testDataLoader = data.DataLoader(testData, batch_size=BATCH_SIZE)

    #
    trainSteps = len(trainDataLoader) // BATCH_SIZE
    valSteps = len(valDataLoader) // BATCH_SIZE

    #
    # print("[INFO] initialising the LeNet model...")
    # model = LeNet(
    #             numChannels=1,
    #             classes=len(trainData.dataset.classes)).to(device)
    # print("[INFO] initialising the RNN model...")
    # model = RNN(classes=len(trainData.dataset.classes)).to(device)
    print("[INFO] initialising the AlexNet model...")
    model = AlexNet(numChannels=3, classes=len(trainData.dataset.classes)).to(device)
    #

    #
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

    #
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    #
    print("[INFO] training the network...")
    startTime = time.time()

    #
    for e in range(0, EPOCHS):
        #
        model.train()

        #
        totalTrainLoss = 0
        totalValLoss = 0

        #
        #
        trainCorrect = 0
        valCorrect = 0

        #
        for (x, y) in trainDataLoader:
            #
            (x, y) = (x.to(device), y.to(device))

            #
            pred = model(x)
            loss = lossFn(pred, y)

            #
            #
            opt.zero_grad()
            loss.backward()
            opt.step()

            #
            #
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        #
        with torch.no_grad():
            #
            model.eval()

            #
            for (x, y) in valDataLoader:
                #
                (x, y) = (x.to(device), y.to(device))

                #
                pred = model(x)
                totalValLoss += lossFn(pred, y)

                #
                valCorrect += (pred.argmax(1) == y).type(
                        torch.float).sum().item()

        #
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        #
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        #
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        #
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:,.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:,.4f}\n".format(
            avgTrainLoss, trainCorrect))

    #
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}".format(
        endTime - startTime))

    #
    print("[INFO] evaluating network...")

    #
    with torch.no_grad():
        #
        model.eval()

        #
        preds = []

        #
        for (x, y) in testDataLoader:
            #
            x = x.to(device)

            #
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    #
    print(metrics.classification_report(testData.targets.cpu().numpy(),
                                        np.array(preds),
                                        target_names=testData.classes))

    #
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_loss"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

    #
    torch.save(model, args["model"])
