if __name__=='__main__':

    import cv2
    import torch
    import random
    import NNclasses
    import wandb
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    from NNclasses import nn_data, Net, CNNet


    root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
    DATA = nn_data(root, batch_size = 16)
    wandb.init(project="CNNet_Sweep2",name = f'lr={config.lr}',entity="kasmello")
    config=wandb.config
    net = CNNet()
    optimizer = optim.AdamW(net.parameters(),lr = config.lr) #adamw algorithm
    epochs = 10
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave = False):
            x,y = batch
            net.zero_grad()
             #sets gradients at 0 after each batch
            output = net(x)
            #calculate how wrong we are
            loss = F.nll_loss(output,y)

            loss.backward()#backward propagation
            optimizer.step()

        with torch.no_grad():
            pred = []
            actual = []

            for batch_v in DATA.all_validation:
                x,y = batch_v
                output = net(x)

                for idx, e in enumerate(output):
                    pred.append(torch.argmax(e))
                    actual.append(y[idx])

            pred = DATA.inverse_encode(pred)
            actual = DATA.inverse_encode(actual)
            print('\n\n')
            output = classification_report(actual,pred, output_dict = True)
            print('\n\n')
            accuracy = output['accuracy']
            precision = output['weighted avg']['precision']
            recall = output['weighted avg']['recall']
            print({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
            wandb.log({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
