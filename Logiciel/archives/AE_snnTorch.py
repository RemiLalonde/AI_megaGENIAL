
import Visualization as vis
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import json

import random
import snntorch as snn
from snntorch import spikegen as spkg
from snntorch import utils
from snntorch import spikeplot as splt

import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
        self.visualization = vis.Visualization()
        layer_sizes = [num_inputs] + hidden_layers

        # Create encoder layers
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            #beta  = random.uniform(0.2, 0.6)
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(snn.Leaky(beta=beta, init_hidden=True, output=(i == len(layer_sizes) - 2), threshold=thresh))  # Add output=True only for the last layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Create decoder layers
        decoder_layers = []
        layer_sizes = list(reversed(hidden_layers)) + [num_output]
        for i in range(len(layer_sizes)-1):
           # beta  = random.uniform(0.2, 0.6)
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            threshold = 20000 if i==1 else thresh
            decoder_layers.append(snn.Leaky(beta=beta, init_hidden=True, output=(i == len(layer_sizes) - 2), threshold=threshold))  # Add output=True only for the last layer
        self.decoder = nn.Sequential(*decoder_layers)

        
    def forward(self, x):
        utils.reset(self.encoder)
        utils.reset(self.decoder)

        mem_rec = []
        spk_rec = []
        mem_rec_1 = []
        spk_rec_1 = []
        for step in range(x.shape[0]):
            spk_mem_tuple  = self.encoder(x)
            spk, mem = spk_mem_tuple[0], spk_mem_tuple[1]

            mem_rec.append(mem)
            spk_rec.append(spk)

        mem_rec = torch.stack(mem_rec)
        spk_rec = torch.stack(spk_rec)

        flattened_samples = mem_rec[-1].view(-1, mem_rec[-1].size(-1))
        flattened_samples_np = flattened_samples.detach().numpy()
        self.kmeans_samples.append(flattened_samples_np)

        for step in range(x.shape[0]):
            # print(spk_rec[step])
            # print(mem_rec[step])
            spk_mem_tuple  = self.decoder(spk_rec)
            spk, mem = spk_mem_tuple[0], spk_mem_tuple[1]

            mem_rec_1.append(mem)
            spk_rec_1.append(spk)

        mem_rec_1 = torch.stack(mem_rec_1)
        spk_rec_1 = torch.stack(spk_rec_1)
        
        return mem_rec_1, spk_rec_1
    
#class Net(nn.Module):
    # def __init__(self, num_inputs, num_outputs, thresh):
    #     super(Net, self).__init__()

    #     # Initialize layers
    #     #self.encoder = create_autoencoder(num_inputs, hidden_layers, num_outputs)
    #     #self.decoder = create_autoencoder(num_outputs, list(reversed(hidden_layers)), num_inputs)
    #     beta = 0.9
    #     self.fc1 = nn.Linear(num_inputs, 8)
    #     self.lif1 = snn.Leaky(beta=beta, threshold=thresh)
    #     self.fc2 = nn.Linear(8, 4)
    #     self.lif2 = snn.Leaky(beta=beta, threshold=thresh)

    #     # Decoder
    #     self.fc3 = nn.Linear(4, 8)
    #     self.lif3 = snn.Leaky(beta=beta, threshold=thresh)
    #     self.fc4 = nn.Linear(8, num_outputs)
    #     self.lif4 = snn.Leaky(beta=beta, threshold=thresh)

    # def forward(self, x): #Dimensions: [Batch,Channels,Width,Length]
    #     utils.reset(self.lif1) #need to reset the hidden states of LIF
    #     utils.reset(self.lif2)
    #     utils.reset(self.lif3)
    #     utils.reset(self.lif4)

    #     mem1 = self.lif1.init_leaky()
    #     mem2 = self.lif2.init_leaky()
    #     mem3 = self.lif3.init_leaky()
    #     mem4 = self.lif4.init_leaky()

    #     mem2_rec = []
    #     spk1_rec = []
    #     spk2_rec = []
    #     mem3_rec = []
    #     spk3_rec = []
    #     mem4_rec = []
    #     spk4_rec = []

    #     for step in range(x.shape[0]):
    #         cur1 = self.fc1(x[step])
    #         spk1, mem1 = self.lif1(cur1, mem1)
    #         cur2 = self.fc2(spk1)
    #         spk2, mem2 = self.lif2(cur2, mem2)

    #         cur3 = self.fc3(spk2)
    #         spk3, mem3 = self.lif3(cur3, mem3)
    #        # print(spk3, mem3, step)
    #         cur4 = self.fc4(spk3)
    #         spk4, mem4 = self.lif4(cur4, mem4)

    #         mem2_rec.append(mem2)
    #         spk1_rec.append(spk1)
    #         spk2_rec.append(spk2)
    #         mem3_rec.append(mem3)
    #         spk3_rec.append(spk3)
    #         mem4_rec.append(mem4)
    #         spk4_rec.append(spk4)

    #     mem2_rec = torch.stack(mem2_rec)
    #     spk1_rec = torch.stack(spk1_rec)
    #     spk2_rec = torch.stack(spk2_rec)
    #     mem3_rec = torch.stack(mem3_rec)
    #     spk3_rec = torch.stack(spk3_rec)
    #     mem4_rec = torch.stack(mem4_rec)
    #     spk4_rec = torch.stack(spk4_rec)
    #     if torch.any(spk4_rec == 1).item():
    #         print("===================================")
    #     out = mem4_rec
    #     return out, spk4_rec


    def train(self, num_epochs, criterion, optimizer, train_set_loader, val_set_loader):
        #normalized_inputs = []
        self.kmeans_samples = []
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_set_loader:
                window_encoded = []
                input, = batch

                optimizer.zero_grad()

                norm_input_mean = input.mean()
                norm_input_std_dev = input.std()
                normalized_input = (input - norm_input_mean) / norm_input_std_dev

                #csv_ino.append(normalized_input)

                #print(normalized_input)
                # for window in normalized_input:
                #     normalized_input_encode = spkg.delta(data=window, threshold=0.3, padding=True, off_spike=True)
                #     window_encoded.append(normalized_input_encode)
                #normalized_encoded_tensor = torch.stack(window_encoded)
                #normalized_input = spikegen.rate(normalized_input, num_steps=1)
                output, spk = net(normalized_input)

                norm_output_mean = output.mean()
                norm_output_std_dev = output.std()
                normalized_output = (output - norm_output_mean) / norm_output_std_dev
                
                loss = criterion(normalized_output, normalized_input)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(train_set_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}" + "\n")
            
            # normalized_output = normalized_output.detach().numpy()
            # normalized_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_input])
            # normalized_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_output])
            num_columns = 4

            #### INPUT + PCA #####
            reshaped_tensor_input = normalized_input.view(-1, normalized_input.size(2))
            data_array_input = reshaped_tensor_input.numpy()

            pca = PCA(n_components=2)
            reduced_data_input = pca.fit_transform(data_array_input)

            num_clusters = 4  # Change this according to your requirement
            kmeans_input = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans_input.fit(reduced_data_input)

            centroids_input = kmeans_input.cluster_centers_
            cluster_labels_input = kmeans_input.labels_
            cluster_labels_input = torch.tensor(cluster_labels_input).view(normalized_input.size(0), normalized_input.size(1))
            
            # #### INPUT - PCA #####
            # reshaped_tensor_input_2 = normalized_input.view(-1, normalized_input.size(2))
            # data_array_input_2 = reshaped_tensor_input_2.numpy()

            # num_clusters = 4  # Change this according to your requirement
            # kmeans_2 = KMeans(n_clusters=num_clusters, random_state=42)
            # kmeans_2.fit(data_array_input_2)

            # centroids_input_2 = kmeans_2.cluster_centers_
            # cluster_labels_input_2 = kmeans_2.labels_
            # cluster_labels_input_2 = torch.tensor(cluster_labels_input_2).view(normalized_input.size(0), normalized_input.size(1))

            self.visualization.saveImage(tensor=normalized_input[0], epoch=epoch, input=True, num_cols=num_columns, step='Train')
            #self.visualization.save_encoding(tensor=normalized_input[0], encoding=normalized_encoded_tensor[0], epoch=epoch, step='Train')
            self.visualization.saveImage(tensor=normalized_output[0], epoch=epoch, input=False, num_cols=num_columns, step='Train')
            self.visualization.saveImageCombined(input_tensor=normalized_input[0], output_tensor=normalized_output[0], epoch=epoch, num_cols=num_columns, step='Train')
            self.visualization.saveImageCombinedOnSame(input_tensor=normalized_input[0], output_tensor=normalized_output[0], epoch=1000, num_cols=num_columns, step='Train')
            self.visualization.save_subplots(input=normalized_input[0], output=normalized_output[0], epoch=epoch, step='Train')
            
            self.visualization.save_KMeans(data=reduced_data_input, centroids=centroids_input, num_clusters=num_clusters, cluster_labels=cluster_labels_input, epoch=epoch, step='Train', input=True)
            # self.visualization.save_KMeans(data=data_array_input_2, centroids=centroids_input_2, num_clusters=num_clusters, cluster_labels=cluster_labels_input_2, epoch=epoch, step='Train', input=False)
            self.visualization.save_raster(data=normalized_input_delta, epoch=epoch, step="Train", input=True)
            # self.visualization.save_KMeans(data=reduced_data_output, centroids=centroids_output, num_clusters=num_clusters, cluster_labels=cluster_labels_output, epoch=epoch, step='Train', input=False)
	    
            # with open("normalized_inputs", "w", newline="") as csvfile:
            #     csv_writer = csv.writer(csvfile)
            #     for normalized_input in normalized_inputs:
            #         csv_writer.writerow(normalized_input)

            val_loss = self.validate(epoch, num_columns, num_epochs, val_set_loader)
            

    def validate(self, epoch, num_columns, num_epochs, val_set_loader):
        with torch.no_grad():
            total_val_loss = 0

            for val_batch in val_set_loader:
                val_input, = val_batch
                norm_val_input_mean = val_input.mean()
                norm_val_input_std_dev = val_input.std()
                normalized_val_input = (val_input - norm_val_input_mean) / norm_val_input_std_dev
                val_output, val_spk = net(normalized_val_input)
                norm_val_output_mean = val_output.mean()
                norm_val_output_std_dev = val_output.std()
                normalized_val_output = (val_output - norm_val_output_mean) / norm_val_output_std_dev
                val_loss = criterion(normalized_val_output, normalized_val_input)
                total_val_loss += val_loss.item()

                # normalized_val_output = normalized_val_output.detach().numpy()
                # normalized_val_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_val_input])
                # normalized_val_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_val_output])
                num_columns = 4

            self.visualization.saveImage(tensor=normalized_val_input[0], epoch=epoch, input=True, num_cols=num_columns, step='Validation')
            self.visualization.saveImage(tensor=normalized_val_output[0], epoch=epoch, input=False, num_cols=num_columns, step='Validation')
            self.visualization.saveImageCombined(input_tensor=normalized_val_input[0], output_tensor=normalized_val_output[0], epoch=epoch, num_cols=num_columns, step='Validation')
            self.visualization.saveImageCombinedOnSame(input_tensor=normalized_val_input[0], output_tensor=normalized_val_output[0], epoch=epoch, num_cols=num_columns, step='Validation')
            self.visualization.save_subplots(input=normalized_val_input[0], output=normalized_val_output[0], epoch=0, step='Validation')
            average_val_loss = total_val_loss / len(val_set_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")
            return average_val_loss

    def test(self, test_set_loader):
        with torch.no_grad():
            total_test_loss = 0
            window_number = 0
            for test_batch in test_set_loader:
                
                test_input, = test_batch
                norm_test_input_mean = test_input.mean()
                norm_test_input_std_dev = test_input.std()
                normalized_test_input = (test_input - norm_test_input_mean) / norm_test_input_std_dev
                test_output, test_spk = net(normalized_test_input)
                norm_test_output_mean = test_output.mean()
                norm_test_output_std_dev = test_output.std()
                normalized_test_output = (test_output - norm_test_output_mean) / norm_test_output_std_dev
                test_loss = criterion(normalized_test_output, normalized_test_input)
                total_test_loss += test_loss.item()

                # normalized_test_output = normalized_test_output.detach().numpy()
                # normalized_test_input = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_test_input])
                # normalized_output = torch.tensor([np.array(row).reshape(128, -1).tolist() for row in normalized_test_output])
                num_columns = 4

                self.visualization.saveImageTestSet(input_tensor=normalized_test_input[0], output_tensor=normalized_test_output[0], window_number=window_number, num_cols=num_columns)
                self.visualization.save_subplotsTest(input=normalized_test_input[0], output=normalized_test_output[0], window_number=window_number)
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
    # data_recordings = [[item for sublist in row for item in sublist] for row in data_recordings]
    # data_recordings_test = [[item for sublist in row for item in sublist] for row in data_recordings_test]
    train_set_loader, val_set_loader, test_set_loader = split_data(data_recordings, data_recordings_test, 0.9, 0.1, 150)
    num_inputs = 4
    hidden_layers = [16, 4]
    num_outputs = 4
    beta = 0.3
    thresh = 1

    net = Net(num_inputs, hidden_layers, num_outputs, thresh, beta)
    print(net)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    net.train(60, criterion, optimizer, train_set_loader, val_set_loader)

    net.test(test_set_loader)

    model_info = {}
    model_info['state_dict'] = {param_tensor: net.state_dict()[param_tensor] for param_tensor in net.state_dict()}
    model_info['parameters'] = [param.data for param in net.parameters()]

    optimizer_info = {var_name: optimizer.state_dict()[var_name] for var_name in optimizer.state_dict()}

    # Save information to a JSON file using the custom encoder
    with open('model_4_input_AE_SNN_info.json', 'w') as json_file:
        json.dump({'model': model_info, 'optimizer': optimizer_info}, json_file, cls=NumpyEncoder)

    print("Model and optimizer information saved to 'model_4_input_AE_SNN_info.json'")

