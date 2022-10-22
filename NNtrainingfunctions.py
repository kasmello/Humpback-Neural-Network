import torch
import pathlib
import wandb
import ssl
import time
import os
import platform
import requests
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch.optim.lr_scheduler as lsr
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from datetime import datetime
from NNclasses import Net, CNNet
from NNfunctions import find_file, get_model_from_name
from ignite.handlers import EarlyStopping



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
    name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr,weight_decay=wd)  # adamw algorithm

    warmup = round(epochs/4)
    if warmup < 2: warmup=2
    rest = epochs-warmup
   
    if lr_decay=='cosineAN': 
        scheduler = create_lr_scheduler_with_warmup(lsr.CosineAnnealingLR(optimizer,T_max=rest),
                        warmup_start_value = lr/warmup,
                        warmup_end_value = lr,
                        warmup_duration=warmup)
    elif lr_decay=='cosineANW':
        scheduler = create_lr_scheduler_with_warmup(lsr.CosineAnnealingWarmRestarts(optimizer,T_0=5),
                        warmup_start_value = lr/warmup,
                        warmup_end_value = lr,
                        warmup_duration=warmup)
    else:
        lr_decay=None
    pathlib.Path(f'Models/{name}').mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    patience=2
    prev_score = 100
    for epoch in tqdm(range(epochs), position=0, leave=True):
        try:
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
                total_time += pause-start
                if i % (3200//len(x)) == 0 and i > 0:
                    net.eval()
                    check_training_accuracy(DATA, net)
                    
                    wandb.log({'Time taken': round(total_time,2)})
                    
            net.eval()
            final_layer = epoch == epochs - 1
            torch.save(net.state_dict(), f'Models/{name}/{name}_{epoch}.nn')
            validate_model(DATA, net, loss.item(), final_layer)
            if prev_score - loss.item() < 0.01:
                patience -= 1
                print(f'Patience now at {patience}')
            else:
                prev_score = loss.item()
                patience=2
            if patience == 0:
                break
        except KeyboardInterrupt:   
            file_path = f'training_in_progress/{name}_{epoch}_{i}.nn'
            torch.save(net.state_dict(), file_path)
            print(f'Keyboard Interrupt detected. Saving current file as {file_path}')
            break
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
            plt.close()

def load_from_recovery(net_name):
    """
    will check the training_progress_folder to see if any files are in there
    if yes, will ask user if they want to load
    """
    pathlib.Path('training_in_progress').mkdir(parents=True,exist_ok=True)
    architecture = net_name.split('_')[0]
    if not bool(pathlib.Path('training_in_progress').glob(f'{architecture}*.nn')):
        print(f'No supporting models for the f{architecture} architecture detected')
        return None
    print('Several recovery models detected')
    model_path = find_file('training_in_progress',f'{architecture}*.nn')
    return model_path
    
    


def run_model(DATA,net_name,lr,wd,momentum,epochs, optimm='sgd', lr_decay=None):
    try:
        requests.get('https://www.google.com')
    except requests.ConnectionError:
        print('Cannot connect to the internet, disabling online mode for WANDB')
        os.environ["WANDB_MODE"]='dryrun'
    # recovery_model_path = load_from_recovery(net_name)
    
    params_str = ''
    if lr:
        params_str += f' lr={lr}'
    if wd:
        params_str += f' wd={wd}'
    if momentum:
        params_str += f' momentum={momentum}'
    if optimm:
        params_str += f' optimm={optimm}'
    if lr_decay:
        params_str += f' lrdecay={lr_decay}'
    params_str += f' batch_size={DATA.batch_size}'
    special_indicator = ''
    if not wandb.run: #If this ain't a sweep
        special_indicator = '_'
        while '_' in special_indicator:
            special_indicator = input('Input special indicator for the NN name for saving. Blank to leave as default\n')
            if '_' in special_indicator: print("ERROR: cannot use '_' in special indicator\n")
        wandb.init(project=net_name.split('_')[0],name=params_str, entity="kasmello",resume="allow")


    if optimm == 'sgd':
        optimm=optim.SGD

    elif optimm == 'adamw':
        optimm = optim.AdamW
        momentum=None


    model = get_model_from_name(net_name,len(DATA.all_labels))  
    train_nn(DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl=f'{net_name}_{params_str}_{special_indicator}',
                                        momentum=momentum,lr_decay=lr_decay)
