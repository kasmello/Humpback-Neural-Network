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
    wandb.init(resume='allow')
    config=wandb.config

    
    params_str = f"arch = {config.architecture} lr={config.lr} lr_decay={config.lr_decay} wd={config.wd} specgram={config.specgram}\
         pink_noise={config.pink} batch_size={config.batch_size}"
    wandb.init().name = params_str
    DATA = nn_data(root, batch_size = config.batch_size,pink=config.pink,specgram=config.specgram)    
    
    run_model(DATA=DATA, net_name=config.architecture, lr=config.lr, wd=config.wd, epochs=config.epochs, \
                        lr_decay = config.lr_decay, pink=config.pink, momentum=None, specgram=config.specgram)