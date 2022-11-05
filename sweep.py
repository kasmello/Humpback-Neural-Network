if __name__=='__main__':

    import wandb
    from NNtrainingfunctions import run_model
    from NNclasses import nn_data
    import platform


    if platform.system() == 'Darwin':  # MAC
        root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/dirty'
    elif platform.system() == 'Windows':
        root = 'C:/Users/Dylan Loo/Desktop/KARMELTINGS/dirty'
    print('Sweep modo!!!')
    wandb.init(resume="allow")
    config=wandb.config
    DATA = nn_data(root, batch_size = config.batch_size,pink=config.pink)    
    
    run_model(DATA=DATA, net_name=config.architecture, lr=config.lr, wd=config.wd, epochs=config.epochs, \
                        lr_decay = config.lr_decay, pink=config.pink, momentum=None)