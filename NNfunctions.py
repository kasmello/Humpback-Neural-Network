import torch
import wandb
from time import time
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch.optim as optim
import torch.nn as nn
from vit_pytorch import ViT
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report
from NNclasses import nn_data, Net, CNNet


def extract_f1_score(DATA, dict):
    data = [[category, dict[category]['f1-score']] for index, category in DATA.label_dict.items()]
    table = wandb.Table(data=data, columns = ["Label", "F1-score"])
    wandb.log({"F1-chart" : wandb.plot.bar(table, "Label", "F1-score",
                                title="F1-chart")})


def train_pretrained_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
                        loss_f=F.nll_loss, momentum=None, weight_decay=0):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=weight_decay)  # adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave=False):
            net.train()
            x, y = batch
            net.zero_grad()
            # sets gradients at 0 after each batch
            output = F.log_softmax(net(x), dim=1)
            # calculate how wrong we are
            loss = loss_f(output, y)
            loss.backward()  # backward propagation
            optimizer.step()

        net.eval()
        final_layer = epoch == epochs - 1
        validate_model(DATA, net, loss, final_layer)


def train_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
             loss_f=F.nll_loss, momentum=None, weight_decay=0):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=weight_decay)  # adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave=False):
            x, y = batch
            net.zero_grad()
            # sets gradients at 0 after each batch
            if str(net) == 'Net':
                output = net(x.view(-1, 224 * 224))
            else:
                output = net(x)
            # calculate how wrong we are
            loss = loss_f(output, y)
            loss.backward()  # backward propagation
            optimizer.step()
        final_layer = epoch == epochs - 1
        validate_model(DATA, net, loss, final_layer)



def validate_model(DATA, net, loss, final_layer):
    with torch.no_grad():
        pred = []
        actual = []

        for batch_v in DATA.all_validation:
            x, y = batch_v
            if str(net) == 'Net':
                output = net(x.view(-1, 224 * 224))
            else:
                output = net(x)

            for idx, e in enumerate(output):
                pred.append(torch.argmax(e))
                actual.append(y[idx])
        pred = DATA.inverse_encode(pred)
        actual = DATA.inverse_encode(actual)
        print('\n\n')
        output = classification_report(actual, pred, output_dict=True)
        accuracy = output['accuracy']
        precision = output['weighted avg']['precision']
        recall = output['weighted avg']['recall']
        print({'Loss': loss, 'Validation Accuracy': accuracy,
              'Wgt Precision': precision, 'Wgt Recall': recall})
        wandb.log({'Loss': loss, 'Validation Accuracy': accuracy,
                  'Wgt Precision': precision, 'Wgt Recall': recall})
        if final_layer:
            extract_f1_score(DATA, output)
            print(classification_report(actual, pred))

def run_model(DATA,net,lr,weight_decay, epochs):
    if net == 'net':
        train_nn(DATA=DATA, net=Net(), lr=lr, weight_decay=weight_decay, epochs=epochs)
    elif net == 'cnnet':
        train_nn(DATA=DATA, net=CNNet(), lr=lr, weight_decay=weight_decay, epochs=epochs)
    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available()
                                else "cpu")
        model = model.to(device)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23)
        momentum = 0.9
        train_pretrained_nn(DATA=DATA, lr=lr, weight_decay=weight_decay, epochs=epochs, optimizer=optim.SGD, net=model, lbl='ResNet18',
                                            loss_f=F.nll_loss, momentum=momentum)
    elif net == 'vgg16':
        model = models.vgg16(pretrained=True)
        momentum = 0.9
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        model.classifier[6].out_features = 23
        train_pretrained_nn(DATA=DATA, lr=lr, weight_decay=weight_decay, epochs=epochs, optimizer=optim.SGD, net=model, lbl='VGG16',
                                            loss_f=F.nll_loss, momentum=momentum)
    elif net == 'vit':
        v = ViT(
            image_size=224,
            patch_size=16,
            num_classes=23,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            channels=1
        )
        momentum = 0.9
        actual, pred = train_pretrained_nn(DATA=DATA, lr=lr, weight_decay=weight_decay, epochs=epochs, optimizer=optim.SGD, net=v, lbl='ViT',
                                            loss_f=F.nll_loss, momentum=momentum)
