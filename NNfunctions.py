import torch
import wandb
import ssl
from time import time
from tqdm import tqdm
import timm
import platform
import transformclasses
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch.optim.lr_scheduler as lsr
from pydub import AudioSegment 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from NNclasses import Net, CNNet

ssl._create_default_https_context = ssl._create_unverified_context


def extract_f1_score(DATA, dict):
    data = [[category, dict[category]['f1-score']] for index, category in DATA.label_dict.items()]
    table = wandb.Table(data=data, columns = ["Label", "F1-score"])
    wandb.log({"F1-chart" : wandb.plot.bar(table, "Label", "F1-score",
                                title="F1-chart")})

def log_images(images,pred,actual):
    images = [wandb.Image(image) for image in images]
    data = [[pred[i],actual[i],images[i]] for i in range(len(images))]
    table = wandb.Table(data=data, columns = ["Predicted", "Actual",'Spectogram'])
    wandb.log({'Image Table': table})

          
def train_pretrained_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
                        loss_f=F.nll_loss, momentum=None, wd=0,lr_decay=None):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=wd)  # adamw algorithm
    scheduler = lr_decay(optimizer,1)
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        scheduler.step()

        net.eval()
        final_layer = epoch == epochs - 1
        check_training_accuracy(DATA, net)
        validate_model(DATA, net, loss, final_layer)


def train_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
             loss_f=F.nll_loss, momentum=None, wd=0,lr_decay=None):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr,weight_decay=wd)  # adamw algorithm
    scheduler = lr_decay(optimizer,1)
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        scheduler.step()
        final_layer = epoch == epochs - 1
        check_training_accuracy(DATA, net)
        validate_model(DATA, net, loss, final_layer)


def predict(data,net, num_batches=999999999):
    with torch.no_grad():
        pred = []
        actual = []
        for i, batch_v in enumerate(data):
            predicted = []
            label = []
            x, y = batch_v
            if str(net) == 'Net':
                output = net(x.view(-1, 224 * 224))
            else:
                output = net(x)

            for idx, e in enumerate(output):
                predicted.append(torch.argmax(e))
                label.append(y[idx])
            pred.extend(predicted)
            actual.extend(label)
            if i == num_batches-1 or i == len(data)-1:
                return x, predicted, label, pred, actual
        

def check_training_accuracy(DATA,net):
    images, predicted, label, pred, actual = predict(DATA.all_training, net, 125)
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
    images, predicted, label, pred, actual = predict(DATA.all_validation,net)
    predicted = DATA.inverse_encode(predicted)
    label = DATA.inverse_encode(label)
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
        log_images(images,predicted,label)
        extract_f1_score(DATA, output)
        print(classification_report(actual, pred))
        cm = confusion_matrix(actual,pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

def run_through_audio():
    example_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/braindead.wav'
    example_track = AudioSegment.from_wav(example_file_path)
    len_of_database = 10000
    size_of_chunks = 2.75  # in seconds
    for t in range(len_of_database - size_of_chunks + 1):
        start = t * 1000
        end = (t + size_of_chunks) * 1000
        print(f'START: {start} END: {end}')
        segment = example_track[start:end]
        segment.export(
            '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav', format='wav')
        wav_to_spectogram(
            '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav', save=False)
        os.remove(
            '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav')
        

def run_model(DATA,net,lr,wd,epochs,momentum, optimm='sgd', lr_decay=None):
    if optimm == 'sgd':
        optimm=optim.SGD
    elif optimm == 'adamw':
        optimm = optim.AdamW
        momentum=None
    device = torch.device("dml" if platform.system()=='Windows'
                                else "cpu")
    if lr_decay=='cosineAN':
        lr_decay = lsr.CosineAnnealingLR
    if net == 'net':
        model = Net()
        model = model.to(device)
        train_nn(DATA=DATA, net=model, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, momentum=momentum,lr_decay=lr_decay)
    elif net == 'cnnet':
        model = CNNet()
        model = model.to(device)
        train_nn(DATA=DATA, net=model, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, momentum=momentum,lr_decay=lr_decay)
    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23)
        model = model.to(device)
        train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='ResNet18',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)
    elif net == 'vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        model.classifier[6].out_features = 23
        model = model.to(device)
        train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='VGG16',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)
    elif net == 'vit':
        model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=23, in_chans=1)
        model = model.to(device)
        train_pretrained_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='ViT',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)



