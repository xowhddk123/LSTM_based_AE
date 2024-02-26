import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.droupout = config.dropout
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=False)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return (hidden, cell)
    
class Decoder(nn.Module):
    
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers
        self.droupout = config.dropout
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=False)
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, hidden):
        outputs, (hidden, cell) = self.lstm(x, hidden)
        pred = self.fc(outputs)
        # pred = self.relu(pred) 
        return pred, (hidden, cell)
    
class LSTMAutoEncoder(nn.Module):
    
    def __init__(self,config)->None:
        
        super(LSTMAutoEncoder, self).__init__()
        
        self.encoder = Encoder(config)
        
        self.reconstruct_decoder = Decoder(config)
        
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, var_length = input.size()
        
        encoder_hidden = self.encoder(input)
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float)
        hidden = encoder_hidden
        
        for _ in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)[:,inv_idx,:]
        
        return reconstruct_output