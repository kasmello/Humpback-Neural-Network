from lib2to3.pytree import convert
import torch
import pathlib
import wandb
import ssl
import time
import os
import platform
import requests
import random
import numpy as np
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



ssl._create_default_https_context = ssl._create_unverified_context
# torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if platform.system()=='Windows'
                                else "mps")

original_patience = 5

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
    x, y = batch[0].to(device,non_blocking=True), batch[1].to(device,non_blocking=True)
    return x,y

def convert_output_to_prediction(output,x,y):
    pred = []
    actual = []
    images = []
    for idx, e in enumerate(output):
        pred.append(torch.argmax(e))
        actual.append(y[idx])
        images.append(x[idx])
    return pred, actual, images


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

    warmup = 3
    rest = epochs-warmup
   
    if lr_decay=='cosineAN': 
        scheduler = lsr.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0.0000001)
    elif lr_decay=='cosineANW':
        scheduler = lsr.CosineAnnealingWarmRestarts(optimizer,T_0=3,eta_min=0.0000001)
    else:
        lr_decay=None
        scheduler=None
    pathlib.Path(f'Models/{name}').mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    patience=original_patience
    prev_score = 0
    final_epoch = 0
    
    for epoch in tqdm(range(epochs), position=0, leave=True):
        try:
            final_epoch = epoch
            curr_learning_rate = optimizer.param_groups[0]["lr"]
            time_taken_this_epoch = 0
            start = time.time()
            for i, batch in enumerate(tqdm(DATA.all_training, position=0, leave=True)):
                net.train()
                x, y = load_batch_to_device(batch)
                x.requires_grad = True
                net.zero_grad()
                # sets gradients at 0 after each batch
                output = F.log_softmax(net(x), dim=1)
                # calculate how wrong we are
                loss = loss_f(output, y)
                loss_number = loss.item()
                if i % 10 == 0 and i > 0:
                    pred, actual, images = convert_output_to_prediction(output, x, y)
                    check_training_accuracy(DATA,pred,actual)
                    if wandb.run:
                        wandb.log({'T Loss': loss_number, 'Learning Rate': curr_learning_rate})
                loss.backward()  # backward propagation
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
            if lr_decay: scheduler.step()
            end = time.time()
            time_taken_this_epoch += end-start
            total_time += time_taken_this_epoch
            net.eval()
            # torch.save(net.state_dict(), f'Models/{name}/{name}_{epoch}.pth')
            patience, prev_score = validate_model(DATA, net, patience, prev_score)
            if patience == 0:
                print('Patience reached 0, break!')
                break
            if total_time >= 7200:
                print('Model has exceeded an hour of training, ending!')
                break

        except KeyboardInterrupt:   
            break
            # file_path = f'training_in_progress/{name}_{epoch}_{i}.pth'
            # torch.save(net.state_dict(), file_path)
            # print(f'Keyboard Interrupt detected. Saving current file as {file_path}')
            # break
    test_model(DATA,net, final_epoch, name)   
    wandb.finish() 


def predict(data,net, num_batches=999999999):
    start = time.time()
    net.eval()
    with torch.no_grad():
        pred = []
        actual = []
        images = []
        output = []
        for i, batch in enumerate(data):
            x, y = load_batch_to_device(batch)
            percents = F.log_softmax(net(x), dim=1)
            loss = F.nll_loss(percents,y)
            loss_number = loss.item()
            new_pred, new_actual, new_images = convert_output_to_prediction(percents, x, y)
            pred.extend(new_pred)
            actual.extend(new_actual)
            images.extend(new_images)
            output.extend(percents.cpu().numpy())
            if i == num_batches-1:
                break
        end = time.time()
        if wandb.run: wandb.log({'Validation Time': end-start})
        return images, pred, actual, loss_number, output
        

def check_training_accuracy(DATA,pred,actual):
    if wandb.run:
        pred = DATA.inverse_encode(pred)
        actual = DATA.inverse_encode(actual)
        output = classification_report(actual, pred, output_dict=True)
        accuracy = output['accuracy']
        precision = output['weighted avg']['precision']
        recall = output['weighted avg']['recall']
        result_dict = {'T Accuracy': accuracy,
                'T Wgt Precision': precision, 'T Wgt Recall': recall}
        wandb.log(result_dict)

def test_model(DATA, net, final_epoch,name):
    images, pred, actual, loss_number, output = predict(DATA.all_testing,net)
    actual_for_roc = actual
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    class_report = classification_report(actual, pred, output_dict=True)
    

    accuracy = class_report['accuracy']
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']
    f1 = class_report['weighted avg']['f1-score']
    
    result_dict = {'Test Loss': loss_number, 'Test Accuracy': accuracy,
            'Test Wgt Precision': precision, 'Test Wgt Recall': recall, 'Test Wgt F1': f1, 'Total Epochs': final_epoch}
    print(result_dict)
    print(classification_report(actual, pred))
    if wandb.run:
        wandb.log({"ROC" : wandb.plot.pr_curve([x.item() for x in actual_for_roc], output,labels=[*DATA.label_dict.values()])})
        wandb.run.summary['Test Results'] = result_dict
        # wandb.log(result_dict)
        idx_images = random.sample(range(0,len(images)),k=50)
        sample_images = [images[i] for i in idx_images]
        sample_pred = [pred[i] for i in idx_images]
        sample_actual = [actual[i] for i in idx_images]
        log_images(sample_images,sample_pred,sample_actual)
        extract_f1_score(DATA, class_report)
    else:
        print(DATA.all_labels)
        cm = confusion_matrix(actual,pred, labels = DATA.all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = DATA.all_labels)
        disp.plot(include_values=False)
        plt.title('ViT Confusion Matrix')
        plt.xticks(rotation=90)
        plt.show()
    torch.save(net.state_dict(), f'Models/{name}/{name}_{final_epoch}.pth')

def validate_model(DATA, net, patience, prev_score):
    images, pred, actual, loss_number, output = predict(DATA.all_validation,net)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    class_report = classification_report(actual, pred, output_dict=True)
    accuracy = class_report['accuracy']
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']
    f1 = class_report['weighted avg']['f1-score']
    result_dict = {'V Loss': loss_number, 'V Accuracy': accuracy,
            'V Wgt Precision': precision, 'V Wgt Recall': recall, 'V Wgt F1': f1}
    print(result_dict)
    if wandb.run:
        wandb.log(result_dict)
    if f1 - prev_score < 0.002:
        patience -= 1
        print(f'Patience now at {patience}')
    else:
        prev_score = f1
    return patience, prev_score

def load_from_recovery(net_name):
    """
    will check the training_progress_folder to see if any files are in there
    if yes, will ask user if they want to load
    """
    pathlib.Path('training_in_progress').mkdir(parents=True,exist_ok=True)
    architecture = net_name.split('_')[0]
    if not bool(pathlib.Path('training_in_progress').glob(f'{architecture}*.pth')):
        print(f'No supporting models for the f{architecture} architecture detected')
        return None
    print('Several recovery models detected')
    model_path = find_file('training_in_progress',f'{architecture}*.pth')
    return model_path
    
    


def run_model(DATA,net_name,lr,wd,momentum,epochs, pink, specgram,optimm='adamw', lr_decay=None):

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
    if optimm:
        params_str += f' optimm={optimm}'
    if lr_decay:
        params_str += f' lrdecay={lr_decay}'
    if pink:
        params_str += f' pink_noise'
    if specgram:
        params_str += f' specgram'
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
                                        momentum=momentum,lr_decay=lr_decay,pink=pink,specgram=specgram)
