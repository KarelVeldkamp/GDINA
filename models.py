
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from helpers import *


class LogisticSampler(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self, slope=1, slope_growth=0):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factorto multiply temperature by each epoch
        """
        super(LogisticSampler, self).__init__()
        # Gumbel distribution
        self.slope = slope
        self.slope_growth = slope_growth

    def forward(self, pi):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        exponent = (pi -.5) * self.slope
        z = torch.sigmoid(exponent)
        return z


class LogLinearSampler(pl.LightningModule):
    def __init__(self, n_attributes):
        super(LogLinearSampler, self).__init__()

        self.intercept = nn.Parameter(torch.Tensor([.5]))
        self.slopes = nn.Parameter(torch.randn(n_attributes))

    def forward(self, alpha):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        logits = alpha @ self.slopes + self.intercept
        pi = F.sigmoid(logits)

        log_pi = torch.cat((pi.log().unsqueeze(2), (1-pi).log().unsqueeze(2)), dim=2)

        g = self.G.sample(log_pi.shape)


        # sample from gumbel softmax
        exponent = (log_pi + g) / self.temperature
        y = self.softmax(exponent)

        y = y[:, :, 0]

        return y



class GumbelSampler(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self,  temperature, decay):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factor to multiply temperature by each epoch
        """
        super(GumbelSampler, self).__init__()
        # Gumbel distribution

        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=2)
        self.temperature = temperature
        self.temperature_decay = decay

    def forward(self, pi, return_effects=True):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        pi=pi.clamp(.001, .999)

        log_pi = torch.cat((pi.log().unsqueeze(2), (1-pi).log().unsqueeze(2)), dim=2)
        g = self.G.sample(log_pi.shape)

        # sample from gumbel softmax
        exponent = (log_pi + g) / self.temperature
        attributes = self.softmax(exponent)


        attributes = attributes[:, :, 0]


        if return_effects:
            effects = expand_interactions(attributes)
            return effects
        else:
            return attributes



class GumbelSoftmax(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self, temperature, decay, ):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factor to multiply temperature by each epoch
        """
        super(GumbelSoftmax, self).__init__()
        # Gumbel distribution

        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.temperature = temperature
        self.temperature_decay = decay


    def forward(self, log_pi, return_effects=True):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        # sample gumbel variable
        g = self.G.sample(log_pi.shape)
        g = g.to(log_pi)
        # sample from gumbel softmax
        attributes = self.softmax((log_pi + g)/self.temperature)

        if return_effects:
            effects = expand_interactions(attributes)
            return effects
        else:
            return attributes

class SpikeAndExpSampler(pl.LightningModule):
    """
    Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self, beta=1):
        super(SpikeAndExpSampler, self).__init__()
        self.beta = torch.Tensor([beta]).float()

    def forward(self, q):
        #clip the probabilities
        q = torch.clamp(q,min=1e-7,max=1.-1e-7)

        #this is a tensor of uniformly sampled random number in [0,1)
        rho = torch.rand(q.size()).to(q)
        zero_mask = torch.zeros(q.size()).to(q)
        ones = torch.ones(q.size()).to(q)
        beta = self.beta.to(q)


        # inverse CDF

        conditional_log = (1./beta)*torch.log(((rho+q-ones)/q)*(beta.exp()-1)+ones)

        zeta=torch.where(rho >= 1 - q, conditional_log, zero_mask)
        return zeta


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 n_items: int,
                 n_attributes: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        self.dense1 = nn.Linear(n_items, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dense3 = nn.Linear(hidden_layer_size, n_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        x = F.elu(self.dense1(x))
        x = F.elu(self.dense2(x))
        x = F.sigmoid(self.dense3(x))
        return x


class GDINADecoder(pl.LightningModule):
    def __init__(self,
                 Q,
                 link):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(GDINADecoder, self).__init__()

        # expand the Q matrix so that the columns represent effects instead of attributes
        self.Q = torch.Tensor(Q)
        self.Q.requires_grad = False

        self.log_delta = nn.Parameter(torch.rand(self.Q.shape).float())
        self.intercepts = nn.Parameter(-torch.rand(self.Q.shape[0]).unsqueeze(0))

        if link == 'logit':
            self.inv_link = nn.Sigmoid()
        elif link == 'log':
            self.inv_link = torch.exp
        elif link == 'dina':
            pass


    def forward(self, Z) -> torch.Tensor:
        delta = self.log_delta * self.Q
        probs = F.sigmoid(Z @ delta.T + self.intercepts)

        return probs


class GDINA(pl.LightningModule):
    def __init__(self, n_items, n_attributes, dataloader, Q, learning_rate,
                 temperature, decay, link, min_temp, LR_min, T_max, LR_warmup):
        super(GDINA, self).__init__()
        self.n_attributes = n_attributes
        self.n_items = n_items

        #self.embedding = nn.Embedding(len(dataloader.dataset), 2)

        self.encoder = Encoder(n_items, self.n_attributes, self.n_attributes)
        self.sampler = GumbelSampler(temperature, decay)
        #self.sampler = LogLinearSampler(self.n_attributes)
        self.decoder = GDINADecoder(Q, link)

        self.dataloader = dataloader
        self.lr = learning_rate
        self.min_temp = min_temp
        self.LR_min = LR_min
        self.T_max = T_max
        self.LR_warmup = LR_warmup



    def forward(self, X):

        pi = self.encoder(X)
        z = self.sampler(pi)

        x_hat = self.decoder(z)


        return x_hat, pi

    def training_step(self, batch):
        X_hat, pi = self(batch)
        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch, reduction='none')
        bce = torch.mean(bce) * self.n_items

        # Compute the KL divergence for each attribute
        pi = pi.clamp(0.0001, 0.9999)
        kl_per_attribute = pi * torch.log(pi / 0.5) + (1 - pi) * torch.log((1 - pi) / 0.5)

        # Sum over the attributes for each instance in the batch
        kl = kl_per_attribute.sum(dim=1)

        # Take the mean over the batch
        kl = torch.mean(kl)

        kl_div = pi * torch.log(pi / 0.5) + (1 - pi) * torch.log((1 - pi) / 0.5)

        # Sum KL divergence over the latent variables (dimension K) and average over the batch
        kl_div = torch.sum(kl_div, dim=1)  # Sum over K latent variables
        kl_div = torch.mean(kl_div)  # Mean over the batch

        loss = bce + kl_div
        self.log('train_loss', loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])


        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam([
                {'params': self.encoder.parameters(), 'lr': self.lr},
                {'params': self.decoder.log_delta, 'lr': self.lr},
                {'params': self.decoder.intercepts, 'lr': self.lr}
            ],
            amsgrad=True
        )
        # Warmup scheduler (linear warmup)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=self.LR_warmup)

        # Cosine annealing scheduler (after warmup)
        #annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.LR_min  # Customize as per your use case
)


        # Combine the schedulers using SequentialLR
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, annealing_scheduler], milestones=[self.LR_warmup]
        )

        scheduler = {
                    'scheduler':scheduler,
                    #'scheduler':annealing_scheduler,
                    #'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=00),
                    #'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0.005, last_epoch=-1),
                    'interval': 'epoch',  # or 'step'
                    'frequency': 1,
                }
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        self.sampler.temperature = max(self.sampler.temperature * self.sampler.temperature_decay, self.min_temp)