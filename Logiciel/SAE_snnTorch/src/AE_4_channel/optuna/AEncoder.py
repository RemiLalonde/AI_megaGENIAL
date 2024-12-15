import torch.nn as nn
import torch.optim as optim
import optuna as opt
import torch
import snntorch as snn
from snntorch import utils

class AEncoder(nn.Module):
    def __init__(self, trial, input_dim, upper_bound, lower_bound):
        super(AEncoder, self).__init__()
        self.trial = trial
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.n_layers = self.trial.suggest_int('n_layers', lower_bound, upper_bound)
        self.beta = self.trial.suggest_float('beta', 0.3, 0.8)
        self.thresh = self.trial.suggest_float('thresh', 0.5, 1)

        encoder_layers = []
        prev_dim_size = [self.input_dim]
        # encoder_layers.append(nn.Linear(self.input_dim, prev_dim_size[-1]))
        # encoder_layers.append(nn.ReLU())

        for i in range(self.n_layers - 1):
            prev_layer_size = prev_dim_size[-1]
            l_bound = max(int(prev_layer_size)/128, 4)
            u_bound = max(int(prev_layer_size)/4, 4)

            next_layer_size = self.trial.suggest_int(f'n_units_l{i}',
                                                     l_bound,
                                                     u_bound)
            encoder_layers.append(nn.Linear(prev_layer_size, next_layer_size))
            prev_dim_size.append(next_layer_size)

            if not i + 1 == self.n_layers:
                #output_layer = i == (self.n_layers - 2)
                encoder_layers.append(snn.Leaky())

        self.encoder = nn.Sequential(*encoder_layers)

        #Decoder
        decoder_layers = []

        for i in range(len(encoder_layers[::-1])):
            if not isinstance(layer, nn.Linear):
                continue 
            #output_layer = i == (len(encoder_layers[::-1]) - 1)
            layer = encoder_layers[::-1][i]

            if isinstance(layer, nn.Linear):  # Check if the layer is a linear layer
                decoder_layers.append(nn.Linear(layer.out_features, layer.out_features))
                decoder_layers.append(nn.ReLU())

                    # p = self.trial.suggest_float(f'dropout_l{i}', 0.05, 0.2)
            # decoder_layers.append(nn.Dropout(p))
        decoder_layers[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*decoder_layers)
        
    
    def forward(self, x):
        # utils.reset(self.encoder)
        # utils.reset(self.decoder)

        # mem_rec = []
        # spk_rec = []
        # mem_rec_1 = []
        # spk_rec_1 = []

        # for step in range(x.shape[0]):
        #     spk_mem_tuple  = self.encoder(x[step])
        #     spk, mem = spk_mem_tuple[0], spk_mem_tuple[1]
            
        #     mem_rec.append(mem)
        #     spk_rec.append(spk)

    
        # mem_rec = torch.stack(mem_rec)
        # spk_rec = torch.stack(spk_rec)

        # for step in range(x.shape[0]):
        #     spk_mem_tuple  = self.decoder(mem_rec[step])
        #     spk, mem = spk_mem_tuple[0], spk_mem_tuple[1]

        #     mem_rec_1.append(mem)
        #     spk_rec_1.append(spk)

        
        # mem_rec_1 = torch.stack(mem_rec_1)
        # spk_rec_1 = torch.stack(spk_rec_1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
