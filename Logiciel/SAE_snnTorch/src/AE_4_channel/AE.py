import Visualization as vis
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split

import json
import random
import numpy as np

# Importing data
def import_data():
    recordings_path_training = '../../../DatasetGenerator/src/data/preprocess/recordings.json'
    with open(recordings_path_training, 'r') as json_file:
        data_recordings = json.load(json_file)

    recordings_path_test = '../../../DatasetGenerator/src/data/preprocess/recordings_small.json'
    with open(recordings_path_test, 'r') as json_file:
        data_recordings_test = json.load(json_file)

    return data_recordings, data_recordings_test

def split_data(data_recordings, data_recordings_test, train_ratio, validate_ratio, num_of_window_to_test):

    total_size = len(data_recordings)
    train_size = int(total_size*train_ratio)
    validate_size = int(total_size*validate_ratio)

    random.shuffle(data_recordings)
    random.shuffle(data_recordings_test)

    train_set = torch.tensor(data_recordings)
    validate_set = torch.tensor(data_recordings[train_size:train_size + validate_size])
    test_set = torch.tensor(data_recordings_test[:num_of_window_to_test])

    train_set_dataset = TensorDataset(train_set)
    train_set_loader = DataLoader(train_set_dataset, batch_size=32, shuffle=True)

    val_set_dataset = TensorDataset(validate_set)
    val_set_loader = DataLoader(val_set_dataset, batch_size=32, shuffle=True)

    test_set_dataset = TensorDataset(test_set)
    test_set_loader = DataLoader(test_set_dataset, batch_size=1, shuffle=False)

    return train_set_loader, val_set_loader, test_set_loader


def createEncoder(num_inputs, hidden_layers):
    layers = []
    layer_sizes = [num_inputs] + hidden_layers
    print(layer_sizes)
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.ReLU())

    # Remove the last ReLU activation in the decoder

    return nn.Sequential(*layers)

def createDecoder(num_output, hidden_layers):
    layers = []
    layer_sizes = list(reversed(hidden_layers)) + [num_output]
    print(layer_sizes)
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.ReLU())

    # Remove the last ReLU activation in the decoder
    layers[-1] = nn.Sigmoid()
    return nn.Sequential(*layers)


# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        super(Net, self).__init__()
        self.visualization = vis.Visualization()
        # Initialize layers
        self.encoder = createEncoder(num_inputs, hidden_layers)
        self.decoder = createDecoder(num_outputs, hidden_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def train(self, num_epochs, criterion, optimizer, train_set_loader, val_set_loader):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_set_loader:

                input, = batch
                optimizer.zero_grad()

                norm_input_mean = input.mean()
                norm_input_std_dev = input.std()
                normalized_input = (input - norm_input_mean) / norm_input_std_dev

                output = net(normalized_input)

                norm_output_mean = output.mean()
                norm_output_std_dev = output.std()
                normalized_output = (output - norm_output_mean) / norm_output_std_dev

                loss = criterion(normalized_output, normalized_input)


                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        

            average_loss = total_loss / len(train_set_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}" + "\n")

            #print(input.shape)
            normalized_output = normalized_output.detach().numpy()
            unflatten_normalized_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_input])
            unflatten_normalized_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_output])
            num_columns = 4
            self.visualization.saveImage(tensor=unflatten_normalized_input[0], epoch=epoch, input=True, num_cols=num_columns, step='Train')
            self.visualization.saveImage(tensor=unflatten_normalized_output[0], epoch=epoch, input=False, num_cols=num_columns, step='Train')
            self.visualization.saveImageCombined(input_tensor=unflatten_normalized_input[0], output_tensor=unflatten_normalized_output[0], epoch=epoch, num_cols=num_columns, step='Train')
            self.visualization.saveImageCombinedOnSame(input_tensor=unflatten_normalized_input[0], output_tensor=unflatten_normalized_output[0], epoch=epoch, num_cols=num_columns, step='Train')
            self.visualization.save_subplots(input=unflatten_normalized_input[0], output=unflatten_normalized_output[0], epoch=epoch, step='Train')

            self.validate(epoch, num_columns, num_epochs, val_set_loader)

    def validate(self, epoch, num_columns, num_epochs, val_set_loader):
        with torch.no_grad():
            total_val_loss = 0
            for val_batch in val_set_loader:
                val_input, = val_batch
                norm_val_input_mean = val_input.mean()
                norm_val_input_std_dev = val_input.std()
                normalized_val_input = (val_input - norm_val_input_mean) / norm_val_input_std_dev
                val_output = net(normalized_val_input)
                norm_val_output_mean = val_output.mean()
                norm_val_output_std_dev = val_output.std()
                normalized_val_output = (val_output - norm_val_output_mean) / norm_val_output_std_dev
                val_loss = criterion(normalized_val_output, normalized_val_input)
                total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_set_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")

            normalized_output = normalized_val_output.detach().numpy()
            unflatten_normalized_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_val_input])
            unflatten_normalized_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_output])
            num_columns = 4

            self.visualization.saveImage(tensor=unflatten_normalized_input[0], epoch=epoch, input=True, num_cols=num_columns, step='Validation')
            self.visualization.saveImage(tensor=unflatten_normalized_output[0], epoch=epoch, input=False, num_cols=num_columns, step='Validation')
            self.visualization.saveImageCombined(input_tensor=unflatten_normalized_input[0], output_tensor=unflatten_normalized_output[0], epoch=epoch, num_cols=num_columns, step='Validation')
            self.visualization.saveImageCombinedOnSame(input_tensor=unflatten_normalized_input[0], output_tensor=unflatten_normalized_output[0], epoch=epoch, num_cols=num_columns, step='Validation')
            self.visualization.save_subplots(input=unflatten_normalized_input[0], output=unflatten_normalized_output[0], epoch=epoch, step='Validation')

    def test(self, test_set_loader):
        with torch.no_grad():
            total_test_loss = 0
            window_number = 0
            for test_batch in test_set_loader:
                
                test_input, = test_batch
                norm_test_input_mean = test_input.mean()
                norm_test_input_std_dev = test_input.std()
                normalized_test_input = (test_input - norm_test_input_mean) / norm_test_input_std_dev
                test_output = net(normalized_test_input)
                norm_test_output_mean = test_output.mean()
                norm_test_output_std_dev = test_output.std()
                normalized_test_output = (test_output - norm_test_output_mean) / norm_test_output_std_dev
                test_loss = criterion(normalized_test_output, normalized_test_input)
                total_test_loss += test_loss.item()

                normalized_output = normalized_test_output.detach().numpy()
                unflatten_normalized_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_test_input])
                unflatten_normalized_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_output])
                num_columns = 4

                self.visualization.saveImageTestSet(input_tensor=unflatten_normalized_input[0], output_tensor=unflatten_normalized_output[0], window_number=window_number, num_cols=num_columns)
                
                window_number += 1

            average_test_loss = total_test_loss / len(test_set_loader)
        
            print(f"Test Loss: {average_test_loss:.4f}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)
    
        
if __name__ == '__main__':

    data_recordings, data_recordings_test = import_data()
    data_recordings = [[item for sublist in row for item in sublist] for row in data_recordings]
    data_recordings_test = [[item for sublist in row for item in sublist] for row in data_recordings_test]
    train_set_loader, val_set_loader, test_set_loader = split_data(data_recordings, data_recordings_test, 0.8, 0.1, 150)
    
    file_path = "data_recordings_test.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Iterate over the rows in the list
        for row in data_recordings_test:
            # Convert each element to a string and write to the file
            file.write(' '.join(map(str, row)) + '\n\n\n')
            
            # Add a blank line after each array
            file.write('\n')

    print(f"Data has been saved to {file_path}")

    num_inputs = 512
    hidden_layers = [128, 4]
    num_outputs = 512
    net = Net(num_inputs, hidden_layers, num_outputs)
    print(net)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train(30, criterion, optimizer, train_set_loader, val_set_loader)

    net.test(test_set_loader)

    model_info = {}
    model_info['state_dict'] = {param_tensor: net.state_dict()[param_tensor] for param_tensor in net.state_dict()}
    model_info['parameters'] = [param.data for param in net.parameters()]

    optimizer_info = {var_name: optimizer.state_dict()[var_name] for var_name in optimizer.state_dict()}

    # Save information to a JSON file using the custom encoder
    with open('model_4_input_AE_info.json', 'w') as json_file:
        json.dump({'model': model_info, 'optimizer': optimizer_info}, json_file, cls=NumpyEncoder)

    print("Model and optimizer information saved to 'model_4_input_AE_info.json'")
