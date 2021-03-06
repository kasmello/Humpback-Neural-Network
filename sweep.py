if __name__=='__main__':

    import wandb
    from NNfunctions import run_model
    from NNclasses import nn_data
    import platform


    if platform.system() == 'Darwin':  # MAC
        root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
    elif platform.system() == 'Windows':
        root = 'C://Users/Karmel 0481098535/Desktop/Humpback'
    DATA = nn_data(root, batch_size = 16)
    wandb.init(project="Overfit Attempt Sweep",entity="kasmello")
    config=wandb.config
    run_model(DATA=DATA, net=config.architecture, lr=config.lr, wd=config.wd, epochs=config.epochs, \
                        momentum=config.momentum, lr_decay = config.lr_decay)
    