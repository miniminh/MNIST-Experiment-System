import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from MnistModel import MnistModel

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())

smaller_train_data, _ = random_split(mnist_trainset, [0.1, 0.9])
train_data, validation_data = random_split(smaller_train_data, [0.7, 0.3])
smaller_test_data, _ = random_split(mnist_testset, [0.1, 0.9])

from utils import MODEL_SAVE_DIR, JOB_SAVE_DIR, PARAMS_SAVE_DIR, save_progress, save_params, load_params, save_model

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    device = 'cpu'
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits = model.eval()(features)
        y_pred = logits.max(1)[1]

        num_examples += targets.size(0)
        correct_pred += (y_pred == targets).sum()
    return correct_pred.float() / num_examples 

def create_train_job(lr, batch_size, epochs, exp_id, save_model_file=True):
    print(f"running experiment: {exp_id}")

    train_loader = DataLoader(train_data, batch_size, shuffle = True)
    val_loader = DataLoader(validation_data, batch_size, shuffle = False)

    print("creating model")
    model = MnistModel()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        history, model = run_epoch(model, optimizer, train_loader, val_loader)
        # val_loss, val_acc = history
        save_progress(history, exp_id, epoch, epochs)
    if save_model_file: 
        save_model(model, exp_id)
        save_params((lr, batch_size, epochs), exp_id)
    return model

default_param_grid = {
    'lr': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'batch_size': [8, 16, 32, 64, 128, 256],
    'epochs': [10, 50, 100, 200],
}
def create_search_job(lr, batch_size, epochs, exp_id):
    print(lr, batch_size, epochs, exp_id)
    job_param_grid = {}
    if lr == 0: 
        job_param_grid['lr'] = default_param_grid['lr']
    else:
        job_param_grid['lr'] = [lr]

    if batch_size == 0:
        job_param_grid['batch_size'] = default_param_grid['batch_size']
    else: 
        job_param_grid['batch_size'] = [batch_size]
    if epochs == 0:
        job_param_grid['epochs'] = default_param_grid['epochs']
    else:
        job_param_grid['epochs'] = [epochs]

    max_acc = 0
    for grid_lr in job_param_grid['lr']:
        for grid_batch_size in job_param_grid['batch_size']:
            for grid_epoch in job_param_grid['epochs']:
                model = create_train_job(grid_lr, grid_batch_size, grid_epoch, exp_id, save_model_file=False)
                acc = float(compute_accuracy(model, DataLoader(smaller_test_data, 8, shuffle = True)))
                print(grid_lr, grid_batch_size, grid_epoch, acc)
                if acc > max_acc: 
                    save_model(model, exp_id)
                    max_acc = acc
                    params = (grid_lr, grid_batch_size, grid_epoch)
    save_params(params, exp_id)


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))

def run_epoch(model, optimizer, train_loader, val_loader):
    
    for batch in train_loader:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    ## Validation phase
    result = evaluate(model, val_loader)
    epoch_res = model.epoch_end(result)
    return epoch_res, model

def load_model_and_eval(exp_id):
    model = MnistModel()
    model.load_state_dict(torch.load(f'{MODEL_SAVE_DIR}/{exp_id}'))
    model.eval()
    acc = float(compute_accuracy(model, DataLoader(smaller_test_data, 8, shuffle = True)))
    return acc

def get_onnx(exp_id):
    model = MnistModel()
    model.load_state_dict(torch.load(f'{MODEL_SAVE_DIR}/{exp_id}'))
    model.eval()
    model_input = torch.randn(1, 1, 28, 28)
    onnx_program = torch.onnx.export(
        model,
        model_input, 
        f"{MODEL_SAVE_DIR}/model_{exp_id}.onnx",
        export_params=True)
    return onnx_program