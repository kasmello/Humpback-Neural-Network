import torch
import wandb
import ssl
import timm
import platform
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch.optim.lr_scheduler as lsr
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from NNclasses import Net, CNNet


ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cpu" if platform.system()=='Windows'
                                else "mps")

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

def load_batch_to_device(batch):
    x, y = batch[0].to(device), batch[1].to(device)
    return x,y


def train_nn(DATA, **train_options):
    lr=train_options['lr']
    optimizer=train_options['optimizer']
    net = train_options['net']
    epochs=train_options['epochs']
    lbl=train_options['lbl']
    loss_f=F.nll_loss
    momentum=train_options['momentum']
    wd=train_options['wd']
    lr_decay=train_options['lr_decay']

    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr,weight_decay=wd)  # adamw algorithm

    warmup = round(epochs/4)
    if warmup < 2: warmup=2
    rest = epochs-warmup
    if lr_decay: scheduler = create_lr_scheduler_with_warmup(lr_decay(optimizer,T_max=rest),
                        warmup_start_value = lr/warmup,
                        warmup_end_value = lr,
                        warmup_duration=warmup)
    for epoch in tqdm(range(epochs), position=0, leave=True):
        net.train()
        if lr_decay: scheduler(None)
        print({'lr': optimizer.param_groups[0]["lr"]})
        for i, batch in enumerate(tqdm(DATA.all_training, position=0, leave=True)):
            x, y = load_batch_to_device(batch)
            x.requires_grad = True
            net.zero_grad()
            # sets gradients at 0 after each batch
            output = F.log_softmax(net(x), dim=1)
            # calculate how wrong we are
            loss = loss_f(output, y)
            loss.backward()  # backward propagation
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            if i % 50 == 0 and i > 0:
                net.eval()
                check_training_accuracy(DATA, net)
        net.eval()
        final_layer = epoch == epochs - 1
        torch.save(net.state_dict(), f'{name}{epoch}.nn')
        validate_model(DATA, net, loss.detach(), final_layer)
    wandb.finish() 


def predict(data,net, num_batches=999999999):
    with torch.no_grad():
        pred = []
        actual = []
        for i, batch_v in enumerate(data):
            x, y = load_batch_to_device(batch_v)
            output = F.log_softmax(net(x), dim=1)
            for idx, e in enumerate(output):
                pred.append(torch.argmax(e))
                actual.append(y[idx])
            if i == num_batches-1 or i == len(data)-1:
                return x, pred, actual
        

def check_training_accuracy(DATA,net):
    images, pred, actual = predict(DATA.all_training, net, 3)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    print('\n\n')
    output = classification_report(actual, pred, output_dict=True)
    accuracy = output['accuracy']
    precision = output['weighted avg']['precision']
    recall = output['weighted avg']['recall']
    result_dict = {'T Accuracy': accuracy,
            'T Wgt Precision': precision, 'T Wgt Recall': recall}
    # print(result_dict)
    if wandb.run:
        wandb.log(result_dict)


def validate_model(DATA, net, loss, final_layer):
    images, pred, actual = predict(DATA.all_validation,net)
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
    if wandb.run:
        wandb.log(result_dict)
    if final_layer:
        log_images(images,pred[-len(images):],actual[-len(images):])
        extract_f1_score(DATA, output)
        print(classification_report(actual, pred))
        cm = confusion_matrix(actual,pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

def run_model(DATA,net,lr,wd,epochs,momentum, optimm='sgd', lr_decay=None):
    if optimm == 'sgd':
        optimm=optim.SGD

    elif optimm == 'adamw':
        optimm = optim.AdamW
        momentum=None

    if lr_decay=='cosineAN':
        lr_decay = lsr.CosineAnnealingLR

    elif lr_decay=='cosineANW':
        lr_decay = lsr.CosineAnnealingWarmRestarts

    if net == 'net':
        model = Net()
        model = model.to(device)
        lbl=None

    elif net == 'cnnet':
        model = CNNet()
        model = model.to(device)
        lbl=None

    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 25)
        model = model.to(device)
        lbl='ResNet18'

    elif net == 'vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        model.classifier[6].out_features = 25
        model = model.to(device)
        lbl='VGG16'

    elif net == 'vit':
        model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=25, in_chans=1)
        model = model.to(device)
        lbl='ViT'
        
    train_nn(DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl=lbl,
                                        momentum=momentum,lr_decay=lr_decay)