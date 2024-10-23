
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.distributions as dist
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

    def forward(self, probs, return_effects=True):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """

        #pi = pi.clamp(0.001, 0.999)
        #pi = torch.cat((pi.unsqueeze(2), (1 - pi).unsqueeze(2)), dim=2)


        # Define the temperature for the RelaxedOneHotCategorical
        temperature = self.temperature

        # Create the RelaxedOneHotCategorical distribution
        #print(temperature)

        distribution = dist.RelaxedOneHotCategorical(temperature, probs=probs)
        #print(torch.round(distribution.rsample(), decimals=2))
        # Sample from the distribution
        attributes = distribution.rsample()
        #attributes = attributes[:, :, 0]

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
        self.dense3 = nn.Linear(hidden_layer_size, n_attributes*2) # multiply by tow for two options (0 or 1)

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
        x = self.dense3(x)

        log_pi = x.reshape((x.shape[0], x.shape[1]//2,2)).exp()


        return log_pi # F.sigmoid(x)


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
                 temperature, decay, link, min_temp, LR_min, T_max, LR_warmup, n_iw_samples):
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

        self.n_samples = n_iw_samples





    def forward(self, X):
        logits = self.encoder(X)
        logits = logits.repeat(self.n_samples, 1, 1,1)
        probs = F.softmax(logits, dim=-1)


        att = self.sampler(probs)
        att = att / att.sum(-1, keepdim=True) # make sure probabilities sum to one (sometimes not true due to numerical issues)
        att = att.clamp(1e-5, 1-1e-5)

        eff = expand_interactions(att[:, :,:, 0])


        x_hat = self.decoder(eff)
        pi = F.softmax(logits, dim=-1)





        # logits = self.encoder(X)
        # #logits = logits.repeat(self.n_samples, 1, 1, 1)
        #
        # probs = F.softmax(logits, dim=-1)
        # probs = probs.repeat(self.n_samples, 1, 1, 1)
        #
        # eff_probs = expand_interactions(probs[:, :, :, 0]).clamp(1e-5, 1-1e-5)
        #
        #
        # eff = self.sampler(eff_probs)
        #
        # eff = eff.clamp(1e-5, 1 - 1e-5)
        #
        # x_hat = self.decoder(eff)
        #
        #
        # pi = torch.cat((eff_probs.unsqueeze(-1), 1-eff_probs.unsqueeze(-1)), dim=-1)
        # eff = torch.cat((eff.unsqueeze(-1), 1-eff.unsqueeze(-1)), dim=-1)


        return x_hat, pi, att

    def loss(self, X_hat, z, pi, batch):
        #bce = torch.nn.functional.binary_cross_entropy(X_hat, batch, reduction='none')
        lll = ((batch * X_hat).clamp(1e-7).log() + ((1 - batch) * (1 - X_hat)).clamp(1e-7).log()).sum(-1, keepdim=True)
        #bce = torch.mean(bce) * self.n_items

        # Compute the KL divergence for each attribute
        #pi = pi.clamp(0.0001, 0.9999)


        kl_type = 'concrete'
        if kl_type == 'categorical':

            bce = torch.nn.functional.binary_cross_entropy(X_hat.squeeze(), batch, reduction='none')
            bce = torch.mean(bce) * self.n_items
            # Analytical KL based on categorical distribution
            kl_div = pi * torch.log(pi / 0.5) + (1 - pi) * torch.log((1 - pi) / 0.5)

            # Sum KL divergence over the latent variables (dimension K) and average over the batch
            kl_div = torch.sum(kl_div, dim=1)  # Sum over K latent variables
            kl_div = torch.mean(kl_div)  # Mean over the batch

            loss = (bce + kl_div)
        elif kl_type == 'concrete':

            log_p_theta = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                        probs=torch.ones_like(pi)).log_prob(z).sum(-1)
            log_q_theta_x = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                          probs=pi).log_prob(z).sum(-1)


            kl = (log_q_theta_x - log_p_theta).unsqueeze(-1)  # kl divergence
            # prior = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]), probs=torch.ones_like(pi))
            # z_dist = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]), probs=pi)
            #
            # kl_div = (z_dist.log_prob(z) - prior.log_prob(z)).sum()


            #loss = lll + kl_div
            elbo = lll - kl

            with torch.no_grad():
                weight = (elbo - elbo.logsumexp(dim=0)).exp()
            #
            loss = (-weight * elbo).sum(0).mean()

        return loss, weight

    def training_step(self, batch):
        X_hat, pi, att = self(batch)
        loss, _ = self.loss(X_hat, att, pi, batch)

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

    def fscores(self, batch, n_mc_samples=50):
        data = batch

        if self.n_samples == 1:
            mu, _ = self.encoder(data)
            return mu.unsqueeze(0)
        else:

            scores = torch.empty((n_mc_samples, data.shape[0], self.n_attributes))
            for i in range(n_mc_samples):

                reco, pi, z = self(data)

                loss, weight = self.loss(reco, z, pi, data)
                z = z[:, :, :, 0]

                idxs = torch.distributions.Categorical(probs=weight.permute(1, 2, 0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]


                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1,
                                      idxs_expanded).squeeze().detach()  # Shape [10000, latent dims]
                if self.n_attributes == 1:
                    output = output.unsqueeze(-1)
                scores[i, :, :] = output

            return scores