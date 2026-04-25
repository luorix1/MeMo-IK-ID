# ICORR_Header_Model.py
import torch.nn as nn
# from torchsummary import summary


# This is a cleaner way to handle padding than manual slicing.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Removes the extra padding from the end of a sequence.
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_layers, kernel_size, stride, dilation,
                 dropout=0.2, norm='weight_norm', activation='ReLU'):
        super(TemporalBlock, self).__init__()

        layers = []
        in_channels = n_inputs
        for i in range(number_of_layers):
            # Calculate padding for causality. This ensures the output length is same as input.
            padding = (kernel_size - 1) * dilation

            # Define the convolutional layer
            conv_layer = nn.Conv1d(in_channels, n_outputs, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation)

            # Apply weight normalization if specified
            if norm == 'weight_norm':
                conv_layer = nn.utils.parametrizations.weight_norm(conv_layer)

            layers.append(conv_layer)

            # Add normalization layer if it's not weight_norm (which is a wrapper)
            if norm and norm != 'weight_norm':
                layers.append(getattr(nn, norm)(n_outputs))

            # Add Chomp1d to remove padding and ensure causality
            layers.append(Chomp1d(padding))
            # Add flexible activation function
            layers.append(getattr(nn, activation)())
            # Add dropout
            layers.append(nn.Dropout(dropout))

            # The input channels for the next layer will be the output channels of this one
            in_channels = n_outputs

        self.network = nn.Sequential(*layers)

        # Downsample layer for the residual connection if channel numbers differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # Final activation for the residual connection
        self.relu = getattr(nn, activation)() # Using the same flexible activation
        self.init_weights()

    def init_weights(self):
        for m in self.network:
            # Check if the module is a Conv1d layer
            if isinstance(m, nn.Conv1d):
                # Check if weight_norm is applied by looking for its parameters
                if hasattr(m, 'weight_v'):
                    # Initialize the direction vector 'v'
                    nn.init.xavier_uniform_(m.weight_v)
                    # Initialize the magnitude 'g' to 1
                    nn.init.constant_(m.weight_g, 1.0)
                else:
                    # Standard initialization if no weight_norm
                    nn.init.xavier_uniform_(m.weight)

        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.network(x)
        res = x if self.downsample is None else self.downsample(x)
        # Because Chomp1d ensures output length == input length, no manual slicing is needed
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, number_of_layers_per_block, kernel_size=2,
                 dropout=0.2, dilations=None, norm='weight_norm', activation='ReLU'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_blocks = len(num_channels)
        
        # Default dilations if not provided: [1, 2, 4, 8, ...]
        if dilations is None:
            dilations = [2**i for i in range(num_blocks)]

        for i in range(num_blocks):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, number_of_layers_per_block,
                                        kernel_size, stride=1, dilation=dilation_size,
                                        dropout=dropout, norm=norm, activation=activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the TCN Network (replacing LSTM)
class TCNModel(nn.Module):
    def __init__(self, hyperparameter_config):
        super(TCNModel, self).__init__()
        
        # Configuration parameters
        self.input_size = hyperparameter_config['input_size']  # Number of sensor inputs (6 for each IMU)
        self.output_size = hyperparameter_config['output_size']  # Number of joint angles
        self.num_channels = hyperparameter_config['num_channels']  # Channels per TemporalBlock
        self.kernel_size = hyperparameter_config['kernel_size']  # Kernel size for convolutional layers
        self.number_of_layers = hyperparameter_config['number_of_layers']
        self.dropout = hyperparameter_config['dropout']  # Dropout value
        self.dilations = hyperparameter_config['dilations']  # Dilations for TemporalBlocks
        self.window_size = hyperparameter_config['window_size']  # Get the sequence length
        
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.number_of_layers, self.kernel_size, self.dropout, self.dilations)
        self.linear = nn.Linear(self.num_channels[-1] * self.window_size, self.output_size)
        
        print("\nTCN parameter #: ", sum(p.numel() for p in self.tcn.parameters()))
        print("\nFCNN parameter #: ",sum(p.numel() for p in self.linear.parameters()))
        
        # Print model summary with auto-calculated sequence length
        # summary(self, input_size=(self.input_size, self.window_size))

    def forward(self, x):
        # x shape: (batch_size, input_size, time window size = sequence length)
        y = self.tcn(x)
        # Flatten the output from the TCN layer
        y = y.flatten(start_dim=1) # Shape: (batch_size, num_channels[-1] * sequence_length)
        y = self.linear(y)
        return y
    
class LSTMModel(nn.Module):
    def __init__(self, hyperparameter_config):
        super(LSTMModel, self).__init__()
        self.input_size = hyperparameter_config['input_size']
        self.hidden_dim = hyperparameter_config['lstm_hidden_dim'] # Use a specific LSTM hidden dim
        self.num_layers = hyperparameter_config['lstm_num_layers'] # Use a specific LSTM num layers
        self.output_size = hyperparameter_config['output_size']
        self.dropout = hyperparameter_config.get('dropout', 0.2) # Use dropout from config or default

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers,
                            batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0)
        self.linear = nn.Linear(self.hidden_dim, self.output_size)
        self.init_weights()
        print("LSTM parameter #: ", sum(p.numel() for p in self.lstm.parameters()) + sum(p.numel() for p in self.linear.parameters()))

    def init_weights(self):
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # Initialize weight matrices orthogonally
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param.data, 0)
                # Optional: Initialize forget gate bias to 1 (helps initial learning)
                # LSTM biases are ordered: [input_gate, forget_gate, cell_gate, output_gate]
                # Find the forget gate bias part (second quarter of the bias vector)
                n = param.size(0)
                start, end = n // 4, n // 2
                nn.init.constant_(param.data[start:end], 1.)

        # Initialize linear layer weights
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, input_size, seq_len)
        lstm_out, (hn, cn) = self.lstm(x.transpose(1,2))
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # hn shape: (num_layers, batch_size, hidden_dim)

        # Use the hidden state of the last layer from the last time step
        last_hidden_state = hn[-1] # Shape: (batch_size, hidden_dim)
        z = self.linear(last_hidden_state) # Shape: (batch_size, output_size)
        return z