MODEL_SAVE_DIR = 'models'
JOB_SAVE_DIR = 'jobs'
PARAMS_SAVE_DIR = 'search_params'

import torch 
import csv

def save_progress(history, exp_id, this_epoch, epoch):
    val_loss, val_acc = history
    with open(f'{JOB_SAVE_DIR}/{exp_id}', 'w+') as f:
        f.write(f"{val_loss},{val_acc},{this_epoch},{epoch}")

def save_params(params, exp_id):
    lr, batch_size, epoch = params
    with open(f'{PARAMS_SAVE_DIR}/{exp_id}', 'w+') as f:
        f.write(f"{lr},{batch_size},{epoch}")
        
def load_params(exp_id):
    try:
        with open(f'{PARAMS_SAVE_DIR}/{exp_id}') as f:
            csv_reader = csv.reader(f, delimiter=',')
            lr = 0
            batch_size = 0
            epoch = 0
            print(csv_reader)
            for row in csv_reader:
                lr = float(row[0]) 
                batch_size = float(row[1])
                epoch = int(row[2])
        return lr, batch_size, epoch
    except:
        return None

def save_model(model, exp_id):
    torch.save(model.state_dict(), f'{MODEL_SAVE_DIR}/{exp_id}')
