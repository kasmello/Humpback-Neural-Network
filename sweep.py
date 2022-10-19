if __name__=='__main__':

    import wandb
    from NNtrainingfunctions import run_model
    from NNclasses import nn_data
    import platform


    if platform.system() == 'Darwin':  # MAC
        root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/clean'
    elif platform.system() == 'Windows':
        root = 'C://Users/Karmel 0481098535/Desktop/Humpback'
    print('Sweep modo!!!')
    wandb.init(resume="allow")
    config=wandb.config
    DATA = nn_data(root, batch_size = config.batch_size)    
    
    run_model(DATA=DATA, net_name=config.architecture, lr=config.lr, wd=config.wd, epochs=config.epochs, \
                        momentum=config.momentum, lr_decay = config.lr_decay)
    