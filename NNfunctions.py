import torch
import wandb
import ssl
from time import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from vit_pytorch import ViT
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report
from NNclasses import Net, CNNet

ssl._create_default_https_context = ssl._create_unverified_context


def extract_f1_score(DATA, dict):
    data = [[category, dict[category]['f1-score']] for index, category in DATA.label_dict.items()]
    table = wandb.Table(data=data, columns = ["Label", "F1-score"])
    wandb.log({"F1-chart" : wandb.plot.bar(table, "Label", "F1-score",
                                title="F1-chart")})


def train_pretrained_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
                        loss_f=F.nll_loss, momentum=None, wd=0):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=wd)  # adamw algorithm
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
        check_training_accuracy(DATA, net)
        validate_model(DATA, net, loss, final_layer)


def train_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
             loss_f=F.nll_loss, momentum=None, wd=0):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr,weight_decay=wd)  # adamw algorithm
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
        check_training_accuracy(DATA, net)
        validate_model(DATA, net, loss, final_layer)


def predict(data,net, num_batches=999999999):
    with torch.no_grad():
        pred = []
        actual = []
        i = 0
        for batch_v in data:
            x, y = batch_v
            if str(net) == 'Net':
                output = net(x.view(-1, 224 * 224))
            else:
                output = net(x)

            for idx, e in enumerate(output):
                pred.append(torch.argmax(e))
                actual.append(y[idx])
            i += 1
            if i == num_batches:
                break
        return pred, actual

def check_training_accuracy(DATA,net):
    pred, actual = predict(DATA.all_training, net, 125)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    print('\n\n')
    output = classification_report(actual, pred, output_dict=True)
    accuracy = output['accuracy']
    precision = output['weighted avg']['precision']
    recall = output['weighted avg']['recall']
    result_dict = {'T Accuracy': accuracy,
            'T Wgt Precision': precision, 'T Wgt Recall': recall}
    print(result_dict)
    wandb.log(result_dict)


def validate_model(DATA, net, loss, final_layer):
    pred, actual = predict(DATA.all_validation,net)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    print('\n\n')
    output = classification_report(actual, pred, output_dict=True)
    accuracy = output['accuracy']
    precision = output['weighted avg']['precision']
    recall = output['weighted avg']['recall']
    result_dict = {'Loss': loss, 'V Accuracy': accuracy,
            'V Wgt Precision': precision, 'V Wgt Recall': recall}
    print(result_dict)
    wandb.log(result_dict)
    if final_layer:
        extract_f1_score(DATA, output)
        print(classification_report(actual, pred))

def run_model(DATA,net,lr,wd,epochs,momentum):
    if net == 'net':
        train_nn(DATA=DATA, net=Net(), lr=lr, wd=wd, epochs=epochs)
    elif net == 'cnnet':
        train_nn(DATA=DATA, net=CNNet(), lr=lr, wd=wd, epochs=epochs)
    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available()
                                else "cpu")
        model = model.to(device)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23)
        train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optim.SGD, net=model, lbl='ResNet18',
                                            loss_f=F.nll_loss, momentum=momentum)
    elif net == 'vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        model.classifier[6].out_features = 23
        train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optim.SGD, net=model, lbl='VGG16',
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
            dropout=0,
            emb_dropout=0,
            channels=1
        )
        actual, pred = train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optim.SGD, net=v, lbl='ViT',
                                            loss_f=F.nll_loss, momentum=momentum)
