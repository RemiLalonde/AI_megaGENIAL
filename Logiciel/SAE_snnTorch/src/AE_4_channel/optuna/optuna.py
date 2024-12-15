import torch.nn as nn
import torch as t
import optuna as opt
import torch.optim as optim
import AEncoder as A
import json
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import random


def import_data():
    recordings_path_training = '../../DatasetGenerator/src/data/preprocess/recordings.json'
    with open(recordings_path_training, 'r') as json_file:
        data_recordings = json.load(json_file)

    recordings_path_test = '../../DatasetGenerator/src/data/preprocess/recordings.json'
    with open(recordings_path_test, 'r') as json_file:
        data_recordings_test = json.load(json_file)

    return data_recordings, data_recordings_test

def split_data(data_recordings, data_recordings_test, train_ratio, validate_ratio, num_of_window_to_test):

    total_size = len(data_recordings)
    train_size = int(total_size*train_ratio)
    validate_size = int(total_size*validate_ratio)

    random.shuffle(data_recordings)
    random.shuffle(data_recordings_test)

    train_set = t.tensor(data_recordings)
    validate_set = t.tensor(data_recordings[train_size:train_size + validate_size])
    test_set = t.tensor(data_recordings_test[:num_of_window_to_test])

    train_set_dataset = TensorDataset(train_set)
    train_set_loader = DataLoader(train_set_dataset, batch_size=64, shuffle=True)

    val_set_dataset = TensorDataset(validate_set)
    val_set_loader = DataLoader(val_set_dataset, batch_size=64, shuffle=False)

    test_set_dataset = TensorDataset(test_set)
    test_set_loader = DataLoader(test_set_dataset, batch_size=1, shuffle=False)

    return train_set_loader, val_set_loader, test_set_loader

def saveImage(tensor, input, trial_number, num_cols):
    os.makedirs(f'images/trial_{trial_number}', exist_ok=True)
                # Plot each column as a separate graph
    for i in range(num_cols):
        plt.plot(tensor.detach().numpy()[:, i], label=f'Column {i + 1}')

    # Add labels and legend
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.legend()

    # Show the plot
    if input:
        plt.savefig(f'images/trial_{trial_number}/last_input.png')
    else:
        plt.savefig(f'images/trial_{trial_number}/last_output.png')
    plt.close()


def objective(trial, dataloader, validation_loader, input_dim, lr_lower_limit, lr_upper_limit, upper_bound, lower_bound, num_epoch):
    lr = trial.suggest_float("lr", lr_lower_limit, lr_upper_limit, log=False)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    model = A.AEncoder(trial, input_dim, upper_bound, lower_bound)
    print(model)
    #print(model)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(num_epoch):
            total_loss = 0
            for batch in train_set_loader:

                input, = batch
                optimizer.zero_grad()

                norm_input_mean = input.mean()
                norm_input_std_dev = input.std()
                normalized_input = (input - norm_input_mean) / norm_input_std_dev

                output = model(normalized_input)

                norm_output_mean = output.mean()
                norm_output_std_dev = output.std()
                normalized_output = (output - norm_output_mean) / norm_output_std_dev

                loss = criterion(normalized_output, normalized_input)


                loss.backward()
                optimizer.step()
                total_loss += loss.item()

       

            average_loss = total_loss
            print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {average_loss:.4f}" + "\n")


            average_loss = validate(epoch, num_epoch, val_set_loader, criterion, model)
    return average_loss

def train(num_epochs, criterion, optimizer, train_set_loader, val_set_loader, model):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_set_loader:

            input, = batch
            optimizer.zero_grad()

            norm_input_mean = input.mean()
            norm_input_std_dev = input.std()
            normalized_input = (input - norm_input_mean) / norm_input_std_dev

            output = model(normalized_input)

            norm_output_mean = output.mean()
            norm_output_std_dev = output.std()
            normalized_output = (output - norm_output_mean) / norm_output_std_dev

            loss = criterion(normalized_output, normalized_input)


            loss.backward()
            optimizer.step()
            total_loss += loss.item()

       

        average_loss = total_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}" + "\n")
        num_columns = 4


        average_loss = validate(epoch, num_epochs, val_set_loader, criterion, model)
        return average_loss

def validate(epoch, num_epochs, val_set_loader, criterion, model):
    with t.no_grad():
        total_val_loss = 0
        for val_batch in val_set_loader:
            val_input, = val_batch
            norm_val_input_mean = val_input.mean()
            norm_val_input_std_dev = val_input.std()
            normalized_val_input = (val_input - norm_val_input_mean) / norm_val_input_std_dev
            val_output = model(normalized_val_input)
            norm_val_output_mean = val_output.mean()
            norm_val_output_std_dev = val_output.std()
            normalized_val_output = (val_output - norm_val_output_mean) / norm_val_output_std_dev
            val_loss = criterion(normalized_val_output, normalized_val_input)
            total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_set_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")

    return average_val_loss

input_dim = 512
trial_number = 0

study = opt.create_study(direction="minimize")
data_recordings, data_recordings_test = import_data()
data_recordings = [[item for sublist in row for item in sublist] for row in data_recordings]
data_recordings_test = [[item for sublist in row for item in sublist] for row in data_recordings_test]
train_set_loader, val_set_loader, test_set_loader = split_data(data_recordings, data_recordings_test, 0.8, 0.1, 150)
study.optimize(lambda trial: objective(trial, train_set_loader, val_set_loader, input_dim, lr_lower_limit=0.001, lr_upper_limit=0.5, upper_bound=6, lower_bound=3, num_epoch=40), n_trials=100)
print(study.best_trial)