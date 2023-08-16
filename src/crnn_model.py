# model.py
import torch
import torch.nn as nn
import numpy as np

class CRNN_Model(nn.Module):
    def __init__(self, num_class:int, msr_size:tuple, rnn_hidden_size:int, dropout:float, tem_fac:list):
        super(CRNN_Model, self).__init__()
        self.num_class = num_class
        self.rnn_hidden_size = rnn_hidden_size
        self.dp = nn.Dropout(p=dropout)
        self.num_freq = msr_size[0]
        self.num_mod = msr_size[1]
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tem_fac = tem_fac
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[0], 1, 1)),
            self.relu
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[1], 1, 1)),
            self.relu
        )
        
        self.cnn3 = nn.Sequential(
            nn.Conv3d(16, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[2], 1, 1)),
            self.relu
        )
        
        
        self.downsample = nn.MaxPool3d((2,2,2))
        
        self.CNNblock = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3
            )
        
        # self.Att = CBAM_Att(channel=4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(20416, 128),
            nn.BatchNorm1d(128),
            self.relu,
            self.dp
            )
        
        # RNN
        self.rnn1 = nn.GRU(input_size=128, 
                            hidden_size=self.rnn_hidden_size,
                            num_layers=3,
                            bidirectional=True, 
                            batch_first=True)
        
        self.layer_norm = nn.LayerNorm([2*self.rnn_hidden_size,int(150/np.product(self.tem_fac))])
        self.maxpool = nn.MaxPool1d(int(150/np.product(self.tem_fac)))
        
        self.fc2 = nn.Linear(self.rnn_hidden_size*2, 1)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        # Apply the CNN layers
        # print(x.shape)
        x = x.unsqueeze(1) 
        # print(x.shape)
        ot = self.CNNblock(x)
        # Flatten ot to have shape (batch_size, -1)
        ot = ot.view(ot.size(0), -1)

        # print(ot.shape)
        # Pass through the first fully connected layer
        ot = self.fc1(ot)
        # print(ot.shape)   
        # After the RNN layer, ot will have shape (batch_size, time_steps, rnn_hidden_size * 2)
        ot, _ = self.rnn1(ot)
        # print("RNN", ot.shape)
        
        ot = self.fc2(ot).squeeze(1).float()
        ot = torch.sigmoid(ot)
        # print("After FC2", ot.shape, ot)
        
        return ot

def initialize_model(num_class, msr_size, rnn_hidden_size, dropout, tem_fac):
    model = CRNN_Model(num_class, msr_size, rnn_hidden_size, dropout, tem_fac)
    return model
