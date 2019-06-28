import sys
import collections

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributed as dist

import modules as custom_nn


def output_to_dist(output, dim=-1):
    z_size = output.size(dim)/2
    mean, log_var = torch.split(output, z_size, dim=dim)
    return torch.distributions.Normal(mean, torch.exp(0.5*log_var))

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
        self.VAEOutput = collections.namedtuple("VAEOutput", ["encoder_out", "decoder_out"])
    
    def forward(self, input, annealing=0):
        
        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out.local_sample)
        assert torch.min(decoder_out) >= 0.
        assert torch.max(decoder_out) <= 1.
        
        return self.VAEOutput(encoder_out, decoder_out)
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.EncoderOutput = collections.namedtuple("EncoderOutput", ["local_dist", "local_sample"])
        
        self.local_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            
            custom_nn.CausalConv1d(201, args.local_z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.local_z_size),
            
            custom_nn.CausalConv1d(args.local_z_size, args.local_z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.local_z_size),
            
            custom_nn.CausalConv1d(args.local_z_size, 2*args.local_z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(2*args.local_z_size),
            
            nn.Conv1d(2*args.local_z_size, 2*args.local_z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2)),
        )
        self.light_dropout = nn.Dropout(0.3)
        
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input is a tensor of batch x time x features
        assert len(input.size()) == 3
        
        local_out = self.local_net(input)
        
        local_dist = output_to_dist(local_out)
        # local sample has siye batch x sample size x time
        local_z_sample = local_dist.rsample()
        
        return self.EncoderOutput(local_dist=local_dist, local_sample=local_z_sample)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(args.local_z_size, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size=1, stride=1),
            nn.Sigmoid(),
            custom_nn.Transpose((1,2)),
        )
    
    def forward(self, input):        
        out = self.fc(input)
        
        return out
