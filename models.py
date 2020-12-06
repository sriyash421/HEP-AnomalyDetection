import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, nodes, dropout, activation, input_size, output_size):
        super(Classifier, self).__init__()
        self.nodes = nodes
        self.input_size = input_size
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU(True)
        self.models = [nn.Sequential(*[nn.Linear(input_size, self.nodes[0]), self.activation_fn, nn.Dropout(p=dropout)])]
        for i in range(len(nodes)-1):
            self.models.append(nn.Sequential(*[nn.Linear(self.nodes[i], self.nodes[i+1]), self.activation_fn, nn.Dropout(p=dropout)]))
        self.models.append(nn.Sequential(*[nn.Linear(self.nodes[-1], output_size), nn.Softmax()]))

        for model in self.models :
            model.apply(self.init_weights)

    def init_weights(self, param):
        if type(param) == nn.Linear:
            torch.nn.init.xavier_uniform_(param.weight.data)

    def forward(self, x):
        outputs = []
        for layer in self.models:
            x = layer(x)
            outputs.append(x)
        outputs = torch.cat(outputs, axis=1)
        return x, outputs

class AutoEncoder(nn.Module):
    def __init__(self, nodes,dropout, activation, input_size):
        super(AutoEncoder, self).__init__()
        self.nodes = nodes
        self.input_size = input_size
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU(True)

        self.modules_downsampling = [
            nn.Linear(input_size, self.nodes[0]), self.activation_fn, nn.Dropout(p=dropout)]
        for i in range(len(nodes)-1):
            self.modules_downsampling.extend([nn.Linear(
                self.nodes[i], self.nodes[i+1]), self.activation_fn, nn.Dropout(p=dropout)])
        
        self.modules_upsampling = []
        for i in range(len(nodes)-1):
            self.modules_upsampling.extend([nn.Linear(
                self.nodes[len(nodes)-1-i], self.nodes[len(nodes)-1-(i+1)]), self.activation_fn, nn.Dropout(p=dropout)])
        self.modules_upsampling.append(nn.Linear(self.nodes[0], input_size))

        self.downsampling = nn.Sequential(*self.modules_downsampling)
        self.upsampling = nn.Sequential(*self.modules_upsampling)
        self.downsampling.apply(self.init_weights)
        self.upsampling.apply(self.init_weights)

    def init_weights(self, param):
        if type(param) == nn.Linear:
            torch.nn.init.xavier_uniform_(param.weight.data)

    def forward(self, x):
        latent_rep = self.downsampling(x)
        return self.upsampling(latent_rep), latent_rep
        