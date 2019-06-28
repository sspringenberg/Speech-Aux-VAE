import collections
import numpy as np
import sys
import os
import io
import ujson

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributed as dist

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab

from audio_utils import to_audio
import librosa
import librosa.display

import PIL.Image
from torchvision.transforms import ToTensor

class VAETrainer:
    
    def __init__(self, args, model, optimizer, train_loader, test_loader, test_dataset, writer):
        self.writer = writer
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.loss_function = args.loss_function
        
        if self.loss_function == 'mi_loss':            
            self.VAELosses = collections.namedtuple("Losses", ["loss", "reconstruction_nll", "prior_nll", "z_prediction_nll", "z_global_entropy", "z_local_entropy"])
        elif self.loss_function == 'kl_loss':
            self.VAELosses = collections.namedtuple("Losses", ["loss", "reconstruction_nll", "prior_nll", "z_local_entropy"])
        elif self.loss_function == 'mi_loss_exp':
            self.VAELosses = collections.namedtuple("Losses", ["loss", "reconstruction_nll", "prior_nll", "z_prediction_nll", "z_entropy", "z_global_entropy", "z_local_entropy"])
        
        self.VAELosses.__new__.__defaults__ = (0,) * len(self.VAELosses._fields)

    def mi_loss(self, input, outputs, beta_kl=10., beta_mi=10., step=0):
        reconstruction = outputs.decoder_out
        global_sample = outputs.encoder_out.global_sample
        local_sample = outputs.encoder_out.local_sample
        
        if self.args.use_cuda:
            prior = torch.distributions.Normal(torch.zeros(local_sample.size()).cuda(), torch.ones(local_sample.size()).cuda())
            data_prop = torch.distributions.Normal(reconstruction, 0.01*torch.ones(reconstruction.size()).cuda())
        else:
            prior = torch.distributions.Normal(torch.zeros(local_sample.size()), torch.ones(local_sample.size()))
            data_prop = torch.distributions.Normal(reconstruction, 0.01*torch.ones(reconstruction.size()))
        prior_ll = torch.mean(prior.log_prob(local_sample))

        # repeat global sample)
        # global_sample_repeated  = global_sample.unsqueeze(1).repeat(1,local_sample.size(1),1)
        global_sample_repeated = global_sample
        
        z_prediction_ll = torch.mean(outputs.predictor_out.log_prob(global_sample_repeated))
        
        reconstruction_ll = -(F.mse_loss(reconstruction, input, size_average=False)/(input.size(0)))/input.size(1)
        
        z_local_entropy = -torch.mean(outputs.encoder_out.local_dist.log_prob(local_sample))
        z_global_entropy = -torch.mean(outputs.encoder_out.global_dist.log_prob(global_sample))
        KL_local_prior = prior_ll + z_local_entropy
        # first term is a cross-entropy from prediction and prior, together with the entropy this is the mutual information
        MI_global_prediction = z_prediction_ll + z_global_entropy
        
        Loss = - reconstruction_ll - beta_kl * KL_local_prior - beta_mi * MI_global_prediction
        
        Losses = self.VAELosses(loss=Loss, reconstruction_nll=-reconstruction_ll, prior_nll=-prior_ll, z_prediction_nll=-z_prediction_ll, z_global_entropy=z_global_entropy, z_local_entropy=z_local_entropy)
        
        return Losses

    def kl_loss(self, input, outputs, beta_kl=10., step=0):
        reconstruction = outputs.decoder_out
        local_sample = outputs.encoder_out.local_sample
        
        if self.args.use_cuda:
            prior = torch.distributions.Normal(torch.zeros(local_sample.size()).cuda(), torch.ones(local_sample.size()).cuda())
            data_prop = torch.distributions.Normal(reconstruction, 0.01*torch.ones(reconstruction.size()).cuda())
        else:
            prior = torch.distributions.Normal(torch.zeros(local_sample.size()), torch.ones(local_sample.size()))
            data_prop = torch.distributions.Normal(reconstruction, 0.01*torch.ones(reconstruction.size()))
        
        reconstruction_ll = -(F.mse_loss(reconstruction, input, size_average=False)/(input.size(0)))/input.size(1)
        
        prior_ll = torch.mean(prior.log_prob(local_sample))
        z_local_entropy = -torch.mean(outputs.encoder_out.local_dist.log_prob(local_sample))
        
        KL_local_prior = prior_ll + z_local_entropy
        
        Loss = - reconstruction_ll - beta_kl * KL_local_prior
        
        Losses = self.VAELosses(loss=Loss, reconstruction_nll=-reconstruction_ll, prior_nll=-prior_ll, z_local_entropy=z_local_entropy)
        
        return Losses
    
    
    def train_epochs(self):
        step = 0
        last_test_losses = [np.Inf]
        if self.args.resume:
            all_train_losses = ujson.load(open("experiments/{}.json".format('train_losses_'+self.args.model_name), 'r'))
            all_test_losses = ujson.load(open("experiments/{}.json".format('test_losses_'+self.args.model_name), 'r'))
        for epoch in range(self.args.num_epochs):
            avg_epoch_losses, step = self.train(step)
            if epoch % self.args.checkpoint_interval == 0:
                # print step
                print('====> Epoch: {} Average train losses'.format(
                    epoch))
                print('-------------------------')
                print avg_epoch_losses
                print('-------------------------')
                
                # self.writer.add_scalars('data/train_losses/', avg_epoch_losses, epoch)
                # for k,v in avg_epoch_losses.iteritems():
                    # self.writer.add_scalar('data/train_losses/'+k, v, epoch)
                
                last_train_losses = avg_epoch_losses['loss']
                
                avg_test_epoch_losses = self.test()
                print('====> Epoch: {} Average test losses'.format(
                    epoch))
                print('-------------------------')
                print avg_test_epoch_losses
                print('-------------------------')
                
                # self.writer.add_scalars('data/test_losses/', avg_test_epoch_losses, epoch)
                # for k,v in avg_test_epoch_losses.iteritems():
                    # self.writer.add_scalar('data/test_losses/'+k, v, epoch)
                
                # if epoch > 5 and avg_test_epoch_losses.loss * 0.2 <= avg_test_epoch_losses.loss - last_test_losses[0]:
                # sys.exit()
                
                last_test_losses = avg_test_epoch_losses['loss']
                
                # recon_img, audio = self.create_reconstruction(2)
                # self.writer.add_image('Reconstruction MSPEC F', recon_img, epoch)
                # self.writer.add_audio('Reconstruction Audio F', audio, epoch, sample_rate=16000)
                
                # recon_img, audio = self.create_reconstruction(80)
                # self.writer.add_image('Reconstruction MSPEC M', recon_img, epoch)
                # self.writer.add_audio('Reconstruction Audio M', audio, epoch, sample_rate=16000)
                
                if not os.path.exists('experiments'):
                    os.makedirs('experiments')
                
                torch.save(self.model.state_dict(), 'experiments/'+self.args.model_name)
                
                if epoch == 0 and not self.args.resume:                    
                    all_train_losses = []
                    all_test_losses = []
                
                all_train_losses.append(avg_epoch_losses)
                all_test_losses.append(avg_test_epoch_losses)
                ujson.dump(all_train_losses,open("experiments/{}.json".format('train_losses_'+self.args.model_name), 'w'))
                ujson.dump(all_test_losses,open("experiments/{}.json".format('test_losses_'+self.args.model_name), 'w'))
    
    def train(self, step):
        self.model.train()
        
        epoch_losses = {}
        
        losses = self.VAELosses()
        
        for name, value in losses._asdict().iteritems():
            epoch_losses[name] = value
        
        batch_amount = 0
        for batch_idx, (data, pers) in enumerate(self.train_loader):
            batch_amount += 1
            self.optimizer.zero_grad()
            
            data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
            
            data = Variable(data)
            
            if self.args.use_cuda:
                data = data.cuda()
            
            data = data.transpose(1,2)
            
            original = data
            
            if self.args.predictive:
                data = F.pad(data, (0,0,1,0), "constant", 1)
                original = F.pad(original, (0,0,0,1), "constant", 1)
            
            outputs = self.model(data, annealing = 0)
            
            if self.loss_function == 'mi_loss':
                losses = self.mi_loss(original, outputs, beta_kl=self.args.beta_kl, beta_mi=self.args.beta_mi, step = step)
            elif self.loss_function == 'kl_loss':
                losses = self.kl_loss(original, outputs, beta_kl=self.args.beta_kl, step = step)
            
            loss = losses.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            for name, value in losses._asdict().iteritems():
                epoch_losses[name] += value.item()
            
            self.optimizer.step()
            step += 1
        
        avg_epoch_losses = {k : round(v/batch_amount,5) for k, v in epoch_losses.iteritems()}
        
        return avg_epoch_losses, step
    
    
    def test(self):
        self.model.eval()
        
        epoch_losses = {}
        losses = self.VAELosses()
        
        for name, value in losses._asdict().iteritems():
            epoch_losses[name] = value
        
        batch_amount = 0
        for batch_idx, (data, pers) in enumerate(self.test_loader):
            
            batch_amount += 1
            data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
            
            with torch.no_grad():
                data = Variable(data)
            if self.args.use_cuda:
                data = data.cuda()
                
            data = data.transpose(1,2)
            
            original = data
            if self.args.predictive:
                data = F.pad(data, (0,0,1,0), "constant", 1)
                original = F.pad(original, (0,0,0,1), "constant", 1)
            
            outputs = self.model(data, annealing = 0)
            
            if self.loss_function == 'mi_loss':
                losses = self.mi_loss(original, outputs, beta_kl=self.args.beta_kl, beta_mi=self.args.beta_mi)
            elif self.loss_function == 'kl_loss':
                losses = self.kl_loss(original, outputs, beta_kl=self.args.beta_kl)
            elif self.loss_function == 'mi_loss_exp':
                losses = self.mi_loss_exp(original, outputs, beta_kl=self.args.beta_kl, beta_mi=self.args.beta_mi)
            
            loss = losses.loss
            
            for name, value in losses._asdict().iteritems():
                epoch_losses[name] += value.item()
        
        avg_epoch_losses = {k : round(v/batch_amount,5) for k, v in epoch_losses.iteritems()}
        
        return avg_epoch_losses
    
    def create_reconstruction(self, indx):
        
        # mel_basis = librosa.filters.mel(16000, 256, n_mels=80, norm=1)
        sr = 16000
        hop_length = int(sr * 0.025)
        n_fft = 1024
        win_length = int(sr * 0.05)
        
        data = self.test_dataset.__getitem__(indx)[0]
        
        data = data / (-80.)
        
        data = Variable(data)
        data = data.unsqueeze(0)
        
        if self.args.use_cuda:
            data = data.cuda()
        
        data = data.transpose(1,2)
        
        original = data
        
        if self.args.predictive:
            data = F.pad(data, (0,0,1,0), "constant", 1)
            original = F.pad(original, (0,0,0,1), "constant", 1)
        
        encoder_outs  = self.model.encoder(data)
        
        if self.loss_function == "mi_loss_exp":
            decoder_out = self.model.decoder(encoder_outs.z_sample)
        else:
            decoder_out = self.model.decoder(encoder_outs.local_sample)
        
        reconstruction = decoder_out.transpose(1,2)
        reconstruction = reconstruction.squeeze(0)
        reconstruction = (reconstruction.data.cpu()).numpy()
        reconstruction = reconstruction * -80.
        
        plt.figure()
        plt.plot([1, 2])
        
        librosa.display.specshow(reconstruction, sr = sr, hop_length = hop_length, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Reconstruction power spectrogram')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        
        reconstruction = librosa.db_to_power(reconstruction)
        
        audio = to_audio(reconstruction, sr=16000,n_fft=1024,hop_t=0.001,win_t=0.025)
        
        audio = 2*((audio-np.min(audio))/(np.max(audio)-np.min(audio)))-1
        
        plt.close()
        
        return image, audio
