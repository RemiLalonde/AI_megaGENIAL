from sklearn.cluster import KMeans, DBSCAN

import Visualization as vis
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import json

import random
import snntorch as snn
from snntorch import spikegen as spkg
from snntorch import utils

from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import csv

# Importing data
def import_data():
    recordings_path_training = '../../../DatasetGenerator/src/data/preprocess/recordings_2min.json'
    with open(recordings_path_training, 'r') as json_file:
        data_recordings = json.load(json_file)

    recordings_path_test = '../../../DatasetGenerator/src/data/preprocess/recordings.json'
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
    test_set = torch.tensor(random.sample(data_recordings_test, num_of_window_to_test))

    # Add the tensor of zeros to the test_set
    train_set_dataset = TensorDataset(train_set)
    train_set_loader = DataLoader(train_set_dataset, batch_size=64, shuffle=True)

    val_set_dataset = TensorDataset(validate_set)
    val_set_loader = DataLoader(val_set_dataset, batch_size=64, shuffle=True)

    test_set_dataset = TensorDataset(test_set)
    test_set_loader = DataLoader(test_set_dataset, batch_size=1, shuffle=True)

    return train_set_loader, val_set_loader, test_set_loader

class Net(nn.Module):
    def __init__(self, num_inputs, hidden_layers, num_output, thresh, beta):
        super(Net, self).__init__()


        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_output = num_output
        self.thresh = thresh
        self.beta = beta

        self.visualization = vis.Visualization()
        layer_sizes = [num_inputs] + hidden_layers

        # Create encoder layers
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(snn.Leaky(beta=beta, init_hidden=True, output=(i == len(layer_sizes) - 2), threshold=thresh))  # Add output=True only for the last layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Create decoder layers
        decoder_layers = []
        layer_sizes = list(reversed(hidden_layers)) + [num_output]
        for i in range(len(layer_sizes)-1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            threshold = 1 if i==1 else thresh
            decoder_layers.append(snn.Leaky(beta=beta, init_hidden=True, output=(i == len(layer_sizes) - 2), threshold=thresh))  # Add output=True only for the last layer
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        spk_mem_tuple  = self.encoder(x)
        spk, mem = spk_mem_tuple[0], spk_mem_tuple[1]
        spk_mem_tuple  = self.decoder(spk)
        spk2, mem2 = spk_mem_tuple[0], spk_mem_tuple[1]
        return mem2, spk2, spk


    def train_model(self, num_epochs, criterion, optimizer, train_set_loader, val_set_loader):
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            signal_encoded = []
            total_loss = 0
            for index, batch in enumerate(train_set_loader):
                window_encoded = []
                ok = []
                input, = batch

                optimizer.zero_grad()

                norm_input_mean = input.mean()
                norm_input_std_dev = input.std()
                normalized_input = (input - norm_input_mean) / norm_input_std_dev

                for window in normalized_input:
                    normalized_input_encode_positive_complete = spkg.delta(data=window, threshold=0.3, padding=True, off_spike=True)
                    normalized_input_encode_positive = spkg.delta(data=window, threshold=0.3, padding=True)
                    normalized_input_encode_negative = spkg.delta(data=window, threshold=-0.3, padding=True)
                    normalized_input_encode_negative = torch.where(normalized_input_encode_negative == 0, torch.tensor(1), torch.tensor(0))
                    normalized_input_encode = torch.cat((normalized_input_encode_positive, normalized_input_encode_negative), dim=1)
                    window_encoded.append(normalized_input_encode)
                    signal_encoded.append(normalized_input_encode)
                    ok.append(normalized_input_encode_positive_complete)
                normalized_encoded_tensor = torch.stack(window_encoded)
                ok_tensor = torch.stack(ok)
                output, spk, _ = net(normalized_encoded_tensor)
                normalized_output = output/torch.max(output)
                loss = criterion(normalized_output, normalized_encoded_tensor)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(train_set_loader)
            train_loss.append(average_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}" + "\n")
            num_columns = 4
            #print(normalized_output)
            self.visualization.saveImage(tensor=normalized_encoded_tensor[0], epoch=epoch, input=True, num_cols=8, step='Train', starting_column=0, positive=True, full=True)
            self.visualization.save_encoding_positive_negative(tensor=ok[0], encoding=normalized_encoded_tensor[0], epoch=epoch, step='Train')
            self.visualization.save_total(data=input[0], encoding=ok[0], positive=normalized_encoded_tensor[0], epoch=epoch, step='Train')
            self.visualization.save_output(output=output[0], epoch=epoch, step='Train', input=False)
            self.visualization.save_output(output=normalized_encoded_tensor[0], epoch=epoch, step='Train', input=True)
            self.visualization.saveImage(tensor=output[0], epoch=epoch, input=False, num_cols=8, step='Train', full=False, starting_column=0, positive=True)
            self.visualization.save_input_output_matrix(signal_encoded[0], epoch, True)
            self.visualization.save_input_output_matrix(output[0], epoch, False)
            if epoch==249:
                with open("inputs.csv", "w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for tensor in signal_encoded[0]:
                        tensor_int = tensor.int().tolist()
                        csv_writer.writerow(tensor_int)
                    csv_writer.writerow('\n\n')
            if epoch==249:
                with open("output.csv", "w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for tensor in output[0]:
                        csv_writer.writerow(tensor.tolist())
                    csv_writer.writerow('\n\n')
            self.validate(epoch, num_columns, num_epochs, val_set_loader, val_loss)
        return train_loss, val_loss
    
    def validate(self, epoch, num_columns, num_epochs, val_set_loader, val_loss_ar):
        with torch.no_grad():
            total_val_loss = 0
            window_encoded = []
            ok = []
            for val_batch in val_set_loader:
                val_input, = val_batch

                norm_val_input_mean = val_input.mean()
                norm_val_input_std_dev = val_input.std()
                normalized_val_input = (val_input - norm_val_input_mean) / norm_val_input_std_dev

                for window in normalized_val_input:
                    normalized_input_encode_positive_complete = spkg.delta(data=window, threshold=0.3, padding=True, off_spike=True)
                    normalized_input_encode_positive = spkg.delta(data=window, threshold=0.3, padding=True)
                    normalized_input_encode_negative = spkg.delta(data=window, threshold=-0.3, padding=True)
                    normalized_input_encode_negative = torch.where(normalized_input_encode_negative == 0, torch.tensor(1), torch.tensor(0))
                    normalized_input_encode = torch.cat((normalized_input_encode_positive, normalized_input_encode_negative), dim=1)
                    window_encoded.append(normalized_input_encode)
                    ok.append(normalized_input_encode_positive_complete)
                normalized_encoded_tensor = torch.stack(window_encoded)

                val_output, val_spk, bot_spk = net(normalized_encoded_tensor)

                reshaped_bot_spk = bot_spk.view(bot_spk.shape[0], -1)
                dbscan = DBSCAN(metric='cosine')
                cluster_labels = dbscan.fit_predict(reshaped_bot_spk.numpy())

                silhouette_avg = silhouette_score(reshaped_bot_spk, cluster_labels)
                sample_sil_values = silhouette_samples(reshaped_bot_spk, cluster_labels)

                print(f'Avg Sil Score: {silhouette_avg}')

                normalized_val_output = val_output/torch.max(val_output)

                val_loss = criterion(normalized_val_output, normalized_encoded_tensor)
                
                total_val_loss += val_loss.item()


            self.visualization.saveImage(tensor=normalized_encoded_tensor[0], epoch=epoch, input=True, num_cols=8, step='Validation', starting_column=0, positive=True, full=True)
            self.visualization.save_encoding_positive_negative(tensor=ok[0], encoding=normalized_encoded_tensor[0], epoch=epoch, step='Validation')
            self.visualization.save_total(data=val_input[0], encoding=ok[0], positive=normalized_encoded_tensor[0], epoch=epoch, step='Validation')
            self.visualization.saveImage(tensor=val_output[0], epoch=epoch, input=False, num_cols=8, step='Validation', full=True, starting_column=0, positive=True)
            average_val_loss = total_val_loss / len(val_set_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss}")
            val_loss_ar.append(average_val_loss)
            return average_val_loss

    def test_model(self, test_set_loader):
        with torch.no_grad():
            total_test_loss = 0
            window_number = 0
            for test_batch in test_set_loader:
                window_encoded = []
                test_input, = test_batch
                norm_test_input_mean = test_input.mean()
                norm_test_input_std_dev = test_input.std()
                normalized_test_input = (test_input - norm_test_input_mean) / norm_test_input_std_dev

                for window in normalized_test_input:
                    normalized_input_encode = spkg.delta(data=window, threshold=0.3, padding=True, off_spike=True)
                    window_encoded.append(normalized_input_encode)
                normalized_encoded_tensor = torch.stack(window_encoded)

                test_output, test_spk, _ = net(normalized_encoded_tensor)
                norm_test_output_mean = test_output.mean()
                norm_test_output_std_dev = test_output.std()
                normalized_test_output = (test_output - norm_test_output_mean) / norm_test_output_std_dev
                test_loss = criterion(normalized_test_output, normalized_encoded_tensor)
                total_test_loss += test_loss.item()
                
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
    bottleneck_data = []
    visualization = vis.Visualization()
    data_recordings, data_recordings_test = import_data()
    train_set_loader, val_set_loader, test_set_loader = split_data(data_recordings['data'], data_recordings_test, 0.9, 0.1, 150)
    num_inputs = 8
    hidden_layers = [128, 64, 32]
    num_outputs = 8
    beta = 0.8
    thresh = 1

    net = Net(num_inputs, hidden_layers, num_outputs, thresh, beta)
    print(net)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_loss, val_loss = net.train_model(250, criterion, optimizer, train_set_loader, val_set_loader)

    torch.save(net.state_dict(), 'SNN_8_inputs.pth')
    visualization.save_train_val_loss(train_loss, val_loss)

    model_info = {}
    model_info['state_dict'] = {param_tensor: net.state_dict()[param_tensor] for param_tensor in net.state_dict()}
    model_info['parameters'] = [param.data for param in net.parameters()]

    optimizer_info = {var_name: optimizer.state_dict()[var_name] for var_name in optimizer.state_dict()}


   # Save information to a JSON file using the custom encoder
    with open('model_4_input_AE_SNN_info.json', 'w') as json_file:
        json.dump({'model': model_info, 'optimizer': optimizer_info}, json_file, cls=NumpyEncoder)

    print("Model and optimizer information saved to 'model_4_input_AE_SNN_info.json'")

