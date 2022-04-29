import torch
import wandb
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report

def train_pretrained_nn(DATA,lr=0.001,optimizer=optim.AdamW,net=None,epochs=10,lbl='',\
                loss_f=F.nll_loss, momentum=None):
    name = str(net)
    if len(str(net))>10:
        name = lbl
    wandb.init(project=name, name=f'lr={lr}',entity="kasmello")
    if momentum:
        optimizer = optimizer(net.parameters(),lr = lr,momentum = momentum)
    else:
        optimizer = optimizer(net.parameters(),lr = lr) #adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave = False):
            net.train()
            x,y = batch
            net.zero_grad()
             #sets gradients at 0 after each batch
            output = F.log_softmax(net(x),dim=1)
            #calculate how wrong we are
            loss = loss_f(output,y)
            loss.backward()#backward propagation
            optimizer.step()
        net.eval()
        actual, pred = validate_model(net,loss)
        if epoch == epochs-1:
            return actual, pred

def train_nn(DATA,lr=0.001,optimizer=optim.AdamW,net=None,epochs=10,lbl='',loss_f=F.nll_loss, momentum=None):
    name = str(net)
    if len(str(net))>10:
        name = lbl
    wandb.init(project=name, name=f'lr={lr}',entity="kasmello")
    if momentum:
        optimizer = optimizer(net.parameters(),lr = lr,momentum = momentum)
    else:
        optimizer = optimizer(net.parameters(),lr = lr) #adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave = False):
            x,y = batch
            net.zero_grad()
             #sets gradients at 0 after each batch
            if str(net)=='Net':
                output = net(x.view(-1,224*224))
            else:
                output = net(x)
            #calculate how wrong we are
            loss = loss_f(output,y)
            loss.backward()#backward propagation
            optimizer.step()
        actual, pred = validate_model(DATA,net,loss)
        if epoch == epochs-1:
            return actual, pred

def validate_model(DATA,net,loss):
    with torch.no_grad():
        pred = []
        actual = []

        for batch_v in DATA.all_validation:
            x,y = batch_v
            if str(net)=='Net':
                output = net(x.view(-1,224*224))
            else:
                output = net(x)

            for idx, e in enumerate(output):
                pred.append(torch.argmax(e))
                actual.append(y[idx])
        pred = DATA.inverse_encode(pred)
        actual = DATA.inverse_encode(actual)
        print('\n\n')
        output = classification_report(actual,pred, output_dict = True)
        accuracy = output['accuracy']
        precision = output['weighted avg']['precision']
        recall = output['weighted avg']['recall']
        print({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
        wandb.log({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
        return actual,pred
