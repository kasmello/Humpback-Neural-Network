import torch
import torch.multiprocessing
import pathlib
import wandb
import ssl
import timm
import time
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
# torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cpu" if platform.system()=='Windows'
                                else "mps")

def extract_f1_score(DATA, dict):
    data = [[category, dict[category]['f1-score']] for category in DATA.label_dict.values()]
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
    pathlib.Path(f'Models/{name}').mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    for epoch in tqdm(range(epochs), position=0, leave=True):
        net.train()
        if lr_decay: scheduler(None)
        print({'lr': optimizer.param_groups[0]["lr"]})
        for i, batch in enumerate(tqdm(DATA.all_training, position=0, leave=True)):
            start = time.time()
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
            pause = time.time()
            total_time += start-pause
            if i % 200 == 0 and i > 0:
                net.eval()
                check_training_accuracy(DATA, net)
            wandb.log({'Time taken': round(total_time,2)})
        net.eval()
        final_layer = epoch == epochs - 1
        
        torch.save(net.state_dict(), f'Models/{name}/{name}{epoch}.nn')
        validate_model(DATA, net, loss.item(), final_layer)
    total_time = round(total_time/60000000000,2)
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
        print(classification_report(actual, pred))
        if wandb.run:
            log_images(images,pred[-len(images):],actual[-len(images):])
            extract_f1_score(DATA, output)
        else:
            print(DATA.all_labels)
            cm = confusion_matrix(actual,pred, labels = DATA.all_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = DATA.all_labels)
            disp.plot()
            plt.show()

def run_model(DATA,net_name,lr,wd,epochs,momentum, optimm='sgd', lr_decay=None):
    if optimm == 'sgd':
        optimm=optim.SGD

    elif optimm == 'adamw':
        optimm = optim.AdamW
        momentum=None

    if lr_decay=='cosineAN':
        lr_decay = lsr.CosineAnnealingLR

    elif lr_decay=='cosineANW':
        lr_decay = lsr.CosineAnnealingWarmRestarts
    else:
        lr_decay = None

    model = get_model_from_name(net_name,len(DATA.all_labels))      
    train_nn(DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl=net_name,
                                        momentum=momentum,lr_decay=lr_decay)

def get_model_from_name(model_name,num_labels):
    if model_name[:3].lower()=='net':
        return Net(num_labels).to(device)

    elif model_name[:5].lower()=='cnnet':
        return CNNet(num_labels).to(device)

    elif model_name[:8].lower()=='resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)
        model = model.to(device)
        return model

    elif model_name[:5].lower()=='vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        #print(model) to look at structure
        model.classifier.add_module('7', nn.ReLU())
        model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
        model.classifier.add_module('9', nn.Linear(1000, num_labels))
        model = model.to(device)
        return model

    elif model_name[:3].lower()=='vit':
        return timm.create_model('vit_tiny_patch16_224',pretrained=True, num_classes=num_labels, in_chans=1).to(device)