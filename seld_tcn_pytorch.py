import torch
import torch.nn as nn
import torch.nn.functional as F

class SELDTCNModel(nn.Module):
    def __init__(self, 
                 data_in, 
                 data_out, 
                 dropout_rate, 
                 nb_cnn2d_filt, 
                 pool_size, 
                 fnn_size,
                 return_embds = False,
                 flatten_embds = False):
        super(SELDTCNModel, self).__init__()

        self.return_embds = return_embds
        self.flatten_embds = flatten_embds
        # Convolutional Layers
        self.conv_layers = nn.ModuleList()
        self.bn2d_layers = nn.ModuleList()
        self.bn1d_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i, pool in enumerate(pool_size):
            self.conv_layers.append(nn.Conv2d(data_in[0] if i == 0 else nb_cnn2d_filt, 
                                              nb_cnn2d_filt, kernel_size=3, padding=1))
            self.bn2d_layers.append(nn.BatchNorm2d(nb_cnn2d_filt))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=(pool, 1)))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # TCN Layers
        self.tcn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        for d in range(10):
            self.tcn_layers.append(nn.Conv1d(nb_cnn2d_filt, 
                                             256, kernel_size=3, padding=2**d, dilation=2**d))
            self.skip_layers.append(nn.Conv1d(256, 128, kernel_size=1, padding=0))
            self.bn1d_layers.append(nn.BatchNorm1d(256))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Final TCN Convolutions
        self.tcn_conv1 = nn.Conv1d(128, 128, kernel_size=1, padding=0)
        self.tcn_conv2 = nn.Conv1d(128, 128, kernel_size=1, padding=0)
        
        # SED Layers
        self.sed_layers = nn.ModuleList()
        self.sed_dropouts = nn.ModuleList()

        self.permute = lambda x: x.permute(0, 2, 1)  # Permute to match (batch, freq, time, channels) 

        self.Q_conv = nn.ModuleList()
        self.K_conv = nn.ModuleList()
        self.V_conv = nn.ModuleList()

        for i in range(2):
            self.Q_conv.append(nn.Conv1d(128,128//8,kernel_size=1,stride=1,padding='same',bias=False))
            self.K_conv.append(nn.Conv1d(128,128//8,kernel_size=1,stride=1,padding='same',bias=False))
            self.V_conv.append(nn.Conv1d(128,128,kernel_size=1,stride=1,padding='same',bias=False))

        for i,fnn in enumerate(fnn_size):
            if i == 0:
                self.sed_layers.append(nn.Linear(128, fnn))
            else:
                self.sed_layers.append(nn.Linear(fnn_size[i-1],fnn_size[i]))
            self.sed_dropouts.append(nn.Dropout(dropout_rate))

        self.sed_out = nn.Linear(fnn_size[-1], data_out[-1])

        if self.return_embds and self.flatten_embds:
            self.flatten = nn.Flatten()

    def forward(self, x):
        # Convolutional Layers
        for conv, bn, pool, drop in zip(self.conv_layers, self.bn2d_layers, self.pool_layers, self.dropout_layers):
            x = F.relu(bn(conv(x)))
            x = pool(x)
            x = drop(x)
        
        # Permute and Reshape
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # TCN Layers with Skip Connections
        skip_connections = []
        for tcn, skip, bn, drop in zip(self.tcn_layers, self.skip_layers, self.bn1d_layers, self.dropout_layers):
            res = x
            x = F.relu(bn(tcn(x)))
            tanh_out = torch.tanh(x)
            sigm_out = torch.sigmoid(x)
            x = tanh_out * sigm_out
            x = drop(x)
            skip_out = skip(x)
            x = res + skip_out
            skip_connections.append(skip_out)
        
        # Sum of Skip Connections
        x = sum(skip_connections)
        x = F.relu(x)

        q = self.Q_conv[0](x)
        k = self.K_conv[0](x)
        v = self.V_conv[0](x)

        score = torch.matmul(torch.transpose(q,1,2),k)
        score = nn.Softmax(dim = -1)(score)
        x = torch.matmul(v,score)
        
        # Final TCN Convolutions
        x = F.relu(self.tcn_conv1(x))
        x = torch.tanh(self.tcn_conv2(x))

        q = self.Q_conv[1](x)
        k = self.K_conv[1](x)
        v = self.V_conv[1](x)

        score = torch.matmul(torch.transpose(q,1,2),k)
        score = nn.Softmax(dim = -1)(score)
        x = torch.matmul(v,score)

        # SED Layers
        x = self.permute(x)

        if self.return_embds:
            if self.flatten_embds:
                x = self.flatten(x)
                return x
            else:
                return x


        for sed, drop in zip(self.sed_layers, self.sed_dropouts):
            x = sed(x)
            x = drop(x)
        
        x = torch.sigmoid(self.sed_out(x))
        return x


if __name__ == "__main__":
    # Example usage
    data_in = (1, 1, 60, 94)  # Example input shape
    data_out = (None, 94, 2)  # Example output shape
    dropout_rate = 0.3
    nb_cnn2d_filt = 128
    pool_size = [4, 3, 3]
    fnn_size = [64, 64]

    model = SELDTCNModel(data_in, 
                         data_out, 
                         dropout_rate, 
                         nb_cnn2d_filt, 
                         pool_size, 
                         fnn_size,
                         return_embds = True,
                         flatten_embds = True)

    sample_data = torch.rand((1,1,60,94))

    total_params = sum(p.numel() for p in model.parameters())
    print(model(sample_data).shape)
    print(total_params)

    # pytorch_module = model.eval()

    # keras_model = nobuco.pytorch_to_keras(
    #     pytorch_module,
    #     args=[sample_data], kwargs=None,
    #     inputs_channel_order=ChannelOrder.TENSORFLOW,
    #     outputs_channel_order=ChannelOrder.TENSORFLOW
    # )

    # print(total_params - keras_model.count_params())
