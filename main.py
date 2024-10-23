from torch.utils.data import DataLoader
from models import *
from data import *
from helpers import *
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
import time

# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# potentially overwrite configurations with command line arguments
if len(sys.argv) > 1:
    cfg["iteration"] = int(sys.argv[1])
    cfg['learning_rate'] = float(sys.argv[2])
    cfg['lr_warmup_epochs'] = int(sys.argv[3])
    cfg['lr_max_epochs'] = int(sys.argv[4])
    cfg['patience'] = int(sys.argv[5])
    cfg['min_delta'] = float(sys.argv[6])
    cfg['gumbel_decay'] = float(sys.argv[7])



delta = torch.Tensor(pd.read_csv(f'true/parameters/delta_{cfg["iteration"]}_{cfg["which_data"]}.csv', index_col=0).values)
intercepts = torch.Tensor(pd.read_csv(f'true/parameters/intercepts_{cfg["iteration"]}_{cfg["which_data"]}.csv', index_col=0).values).T

Q = (delta != 0).int()


attributes = torch.Tensor(pd.read_csv(f'true/parameters/att_{cfg["iteration"]}_{cfg["which_data"]}.csv', index_col=0).values)

data = torch.Tensor(pd.read_csv(f'true/data/data_{cfg["iteration"]}_{cfg["which_data"]}.csv', index_col=0).values)
n_items = data.shape[1]
n_attributes = int(np.log2(Q.shape[1]+1))


if cfg['which_data'] == 'llm':
    link='logit'
elif cfg['which_data'] == 'gdina':
    link='logit'
elif cfg['which_data'] == 'dina':
    link='dina'
elif cfg['which_data'] == 'rrum':
    link='log'



device = torch.device("cpu")
# create pytorch dataset
dataset = MemoryDataset(data, device)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)

model = GDINA(dataloader=train_loader,
             n_items=n_items,
             n_attributes=n_attributes,
             Q=Q,
             learning_rate=cfg['learning_rate'],
             temperature=cfg['gumbel_temperature'],
             decay=cfg['gumbel_decay'],
             link=link,
             min_temp=cfg['gumbel_min_temp'],
             T_max = cfg['lr_max_epochs'],
             LR_min= cfg['lr_min'],
             LR_warmup=cfg['lr_warmup_epochs'],
             n_iw_samples=cfg['n_iw_samples']
             )


model.to(device)


if os.path.exists('logs/all/version_0/metrics.csv'):
    os.remove('logs/all/version_0/metrics.csv')


logger = CSVLogger("logs", name='all', version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  min_epochs=cfg['min_epochs'],
                  logger=logger,
                  callbacks=[
                  EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'],
                                mode='min')],
                  accelerator=device.type,
                  detect_anomaly=False)
start = time.time()

trainer.fit(model)
runtime = time.time() - start

print(f'runtime: {runtime}')

# compute the estimated conditional probabilities and the posterior probabilities on the test data
test_data = next(iter(test_loader))


#pred_class = (model.encoder(test_data) > .5).float().detach().numpy()
pred_class = (model.fscores(test_data).mean(0) > .5).float().detach().numpy()

print(expand_interactions(model.fscores(test_data).mean(0)).mean(1))
print(np.rint(expand_interactions(model.fscores(test_data).mean(0))).mean(1))


print(f'temperature: {model.sampler.temperature}')



#acc = (expand_interactions(attributes).detach().numpy()==pred_class).mean()
acc = (attributes.detach().numpy()==pred_class).mean()

print(f'accruracy: {acc}')




est_delta = (model.decoder.log_delta * model.decoder.Q).detach().numpy()
est_intercepts = model.decoder.intercepts.detach().numpy()

delta = delta.detach().numpy()

intercepts = intercepts.detach().numpy()

n_effects = n_attributes**2-1


if len(sys.argv) > 1:
    mse_delta = MSE(est_delta[delta!=0], delta[delta!=0])
    mse_intercept = MSE(est_intercepts, intercepts)
    metrics = [acc, mse_delta, mse_intercept, runtime]
    with open(f'results/metrics/metrics_{cfg["iteration"]}_{cfg["learning_rate"]}_'
              f'{cfg["lr_warmup_epochs"]}_{cfg["lr_max_epochs"]}_{cfg["patience"]}_{cfg["min_delta"]}_'
              f'{cfg["gumbel_decay"]}.txt', 'w') as f:
        for metric in metrics:
            f.write(f"{metric}\n")
else:
    # remove any old plots
    old_plots = glob.glob(f'./figures/simfit/{cfg["which_data"]}/*')
    for plot in old_plots:
        os.remove(plot)

    for effect in range(0,n_effects-1):
        plt.figure()
        mse = MSE(est_delta[:,effect], delta[:,effect])
        plt.scatter(y=est_delta[:,effect], x=delta[:,effect])
        plt.plot(delta[:,effect], delta[:,effect])
        # for i, x in enumerate(ai_true):
        #    plt.text(ai_true[i], ai_est[i], i)
        plt.title(f'Parameter estimation plot: delta {effect + 1}, MSE={round(mse, 4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/simfit/{cfg["which_data"]}/param_est_plot_delta{effect + 1}.png')

    plt.figure()
    mse = MSE(est_delta[delta!=0], delta[delta!=0])
    plt.scatter(y=est_delta[delta!=0], x=delta[delta!=0])
    plt.plot(delta[delta!=0], delta[delta!=0])
    # for i, x in enumerate(ai_true):
    #    plt.text(ai_true[i], ai_est[i], i)
    for i in range(est_delta.shape[0]):
        for j in range(est_delta.shape[1]):
            if delta[i,j] != 0:
                plt.text(delta[i,j], est_delta[i,j], str(f'{i}{j}'), fontsize=8, ha='right', va='bottom')

    plt.title(f'Parameter estimation plot: delta, MSE={round(mse, 4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/simfit/{cfg["which_data"]}/param_est_plot_delta.png')


    plt.figure()
    plt.scatter(y=est_intercepts, x=intercepts)
    plt.plot(intercepts,intercepts)
    mse = MSE(est_intercepts, intercepts)
    plt.title(f'Parameter estimation plot: d, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/simfit/{cfg["which_data"]}/param_est_plot_intercepts.png')


    # plot training loss
    plt.figure()
    logs = pd.read_csv(f'logs/all/version_0/metrics.csv')
    plt.plot(logs['epoch'], logs['train_loss'])
    plt.title('Training loss')
    plt.savefig(f'./figures/simfit/{cfg["which_data"]}/training_loss.png')


    # plot learning rate
    plt.figure()
    logs = pd.read_csv(f'logs/all/version_0/metrics.csv')
    plt.plot(logs['epoch'], logs['lr'])
    plt.title('Learning rate')
    plt.savefig(f'./figures/simfit/{cfg["which_data"]}/learning_rate.png')


