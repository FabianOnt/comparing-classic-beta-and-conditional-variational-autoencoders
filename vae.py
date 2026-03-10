import torch
import numpy as np

import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal as MVN
from tqdm.notebook import tqdm

from  torch.nn import Module
from torch.nn.functional import one_hot
from functools import singledispatchmethod

from IPython.display import clear_output

from torch.utils.data import DataLoader

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        General Early Stopping procedure.

        patience: how many epochs to wait after last improvement
        min_delta: minimum improvement in val_loss to qualify as better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class VAE(Module):
    """
    Variational Autoencoder (VAE) implementation.

    This class can represent:
      - A classic VAE (default).
      - A β-VAE if `beta ≠ 1.0`.
      - A Conditional VAE if `conditional=True`.

    It supports training with early stopping, ELBO evaluation,
    latent prior estimation, posterior sampling, and visualization
    of optimization history.
    """
    def __init__(self, latent_dims:int, encoder: Module, mu: Module, logvar: Module, decoder: Module, 
                 alpha: float, beta: float = 1.0, conditional: bool = False, categorical_conditioned: bool = False,
                 one_hot_encode: bool = False, num_classes: int = None):
        """
        Initialize the VAE model.

        Args:
            latent_dims (int): Dimensionality of the latent space.
            encoder (Module): Encoder network mapping inputs to hidden features.
            mu (Module): Module projecting encoder features to latent mean.
            logvar (Module): Module projecting encoder features to latent log-variance.
            decoder (Module): Decoder network reconstructing data from latent vectors.
            alpha (float): Precision parameter for reconstruction likelihood.
            beta (float, optional): Weight for KL divergence term (β-VAE). Default: 1.0.
            conditional (bool, optional): Whether to use conditional inputs. Default: False.
            categorical_conditioned (bool, optional): Whether the model is class conditioned. Default: False.
            one_hot_encode (bool, optional): Whether to apply one-hot encoding to the conditioned variable. 
                It rquieres categorical_conditioned set to Ture. Default: False.
            num_classes (int, optional): Total number of classes. It rquieres categorical_conditioned 
                and one_hot_encode set to Ture. Default: None.
        """
        super().__init__()

        self.latent_dims = latent_dims
        self.encoder = encoder
        self.mu = mu
        self.logvar = logvar
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta

        self.conditional = conditional
        self.categorical_conditioned = categorical_conditioned
        self.one_hot_encode = one_hot_encode
        self.num_classes = num_classes

        self.history: dict[str,list] | None = None
        self.device: str = "cpu"

        self.__latent_mean = None
        self.__latent_cov = None

        if categorical_conditioned: 
            assert self.conditional 
        elif one_hot_encode:
            assert num_classes is not None

        self.reset_history()

    def __one_hot_encode(self, y):
        return one_hot(y, num_classes=self.num_classes).float()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, size:int=1):
        """
        Apply the reparameterization trick to sample latent variables.

        Args:
            mu (Tensor): Mean of latent Gaussian distribution.
            logvar (Tensor): Log-variance of latent Gaussian distribution.
            size (int, optional): Number of latent samples per input. Default: 1.

        Returns:
            Tensor: Sampled latent variables.
        """
        std = torch.exp(0.5 * logvar)
        if size == 1:
            eps = torch.randn_like(std, device=self.device)
            return mu + std * eps
        else:
            eps = torch.randn(size,*std.shape, device=self.device)
            return mu.unsqueeze(0) + std.unsqueeze(0) * eps

    def encode(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Encode input data into latent distribution parameters.

        Args:
            x (Tensor): Input data.
            y (Tensor, optional): Conditional labels (if conditional VAE).

        Returns:
            (Tensor, Tensor): Mean (mu) and log-variance (logvar) of latent distribution.
        """
        if self.conditional:
            if self.one_hot_encode:
                w = torch.cat([x,self.__one_hot_encode(y)],dim=-1)
            else:
                w = torch.cat([x,y],dim=-1)
        else:
            w = x
        h = self.encoder(w)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor, y: torch.Tensor = None):
        """
        Encode input data into latent distribution parameters.

        Args:
            x (Tensor): Input data.
            y (Tensor, optional): Conditional labels (if conditional VAE).

        Returns:
            (Tensor, Tensor): Mean (mu) and log-variance (logvar) of latent distribution.
        """
        if self.conditional:
            if self.one_hot_encode:
                w = torch.cat([z,self.__one_hot_encode(y)],dim=-1)
            else:
                w = torch.cat([z,y],dim=-1)
        else:
            w = z
        x_recon = self.decoder(w)
        return x_recon

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass through the VAE.

        Args:
            x (Tensor): Input data.
            y (Tensor, optional): Conditional labels (if conditional VAE).

        Returns:
            tuple: (Reconstructed data, latent mean, latent log-variance).
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar
    
    def cuda(self):
        """
        Move model to CUDA device.

        Returns:
            VAE: Model on CUDA.
        """
        self.device = "cuda"
        super().cuda()
        return self
    
    def cpu(self):
        """
        Move model to CPU.

        Returns:
            VAE: Model on CPU.
        """
        self.device = "cpu"
        super().cpu()
        return self

    def to(self, device:str):
        """
        Move model to a specified device.

        Args:
            device (str): Target device ("cpu" or "cuda").

        Returns:
            VAE: Model on target device.
        """
        self.device = device
        super().to(device)
        return self

    @singledispatchmethod
    def estimate_prior(self, dataloader: DataLoader):
        """
        Estimate empirical prior distribution of latent space from data.
        Updates self.__latent_mean and self.__latent_cov with empirical estimates.

        Args:
            dataloader (DataLoader): Loader providing data batches.

        """
        if self.categorical_conditioned:
            self.__latent_mean = {}
            self.__latent_cov = {}
            a = {_:torch.zeros(self.latent_dims, device=self.device) for _ in range(self.num_classes)}
            b = {_:torch.zeros((self.latent_dims, self.latent_dims), device=self.device) for _ in range(self.num_classes)}
            n_samples = {_:0 for _ in range(self.num_classes)}
            for x,y in dataloader:
                if (np.array(list(n_samples.values())) >= 500).all():
                    break
                for c in range(self.num_classes):
                    if n_samples[c] >= 500:
                        break
                    z = self.encode(x[y==c],y[y==c])[0]
                    a[c] += z.sum(dim=0)
                    b[c] += z.T @ z
                    n_samples[c] += z.shape[0]
            for c in range(self.num_classes):
                self.__latent_mean.update({c: a[c] / n_samples[c]})
                self.__latent_cov.update({c: (b[c] - torch.outer(a[c],a[c]) / n_samples[c]) / (n_samples[c] - 1)})
        else:
            a = torch.zeros((self.latent_dims,), device=self.device)
            b = torch.zeros((self.latent_dims, self.latent_dims), device=self.device)
            n_samples = 0
            for batch in dataloader:
                if n_samples >= 500:
                    break
                if self.conditional:
                    z = self.encode(*batch)[0]
                else:
                    z = self.encode(batch)[0]
                a += z.sum(dim=0)
                b += z.T @ z
                n_samples += z.shape[0]
            self.__latent_mean = a / n_samples
            self.__latent_cov = (b - torch.outer(a,a) / n_samples) / (n_samples - 1)

    @estimate_prior.register
    def _(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Estimate prior distribution directly from a batch of tensors.
        Updates self.__latent_mean and self.__latent_cov with empirical estimates.

        Args:
            x (Tensor): Input data.
            y (Tensor, optional): Conditional labels (if conditional VAE).
        """
        z = self.decode(x, y)
        self.__latent_mean = z.mean(dim=0)
        self.__latent_cov = torch.cov(z.T)

    def elbo(self, x:torch.Tensor, y:torch.Tensor=None, n_samples:int=20):
        """
        Compute Evidence Lower Bound (ELBO) for a batch.

        Args:
            x (Tensor): Input data.
            y (Tensor, optional): Conditional labels (if conditional VAE).
            n_samples (int, optional): Number of latent samples for Monte Carlo estimate.

        Returns:
            Tensor: Scalar ELBO estimate.
        """
        mu, logvar = self.encode(x,y)
        sigma = torch.exp(0.5 * logvar)
        z = self.reparameterize(mu, logvar, n_samples)
        if self.conditional:
            if self.one_hot_encode:
                y_repeated = y.repeat(n_samples,1)
            else:
                y_repeated = y.repeat(n_samples,1,1)
        else:
            y_repeated = None
        expected_log_prob = MVN(
                loc=torch.zeros(x.shape[-1], device=self.device),
                covariance_matrix=self.alpha*torch.eye(x.shape[-1], device=self.device)
        ).log_prob(x.unsqueeze(0) - self.decode(z,y_repeated)).mean(dim=0).sum()

        kl_divergence = 0.5 * (sigma*sigma + mu*mu - 2*torch.log(sigma) - 1).sum()
        
        elbo_estimator = expected_log_prob - self.beta * kl_divergence
        del expected_log_prob, kl_divergence, z, mu, sigma
        return elbo_estimator 
    
    
    def fit(self, optim: torch.optim, train_loader: DataLoader, val_loader: DataLoader = None,
            n_samples_latent: int = 25, epochs: int = 3, patience: int = 5,
            min_delta: float = 1e-4, reset_history: bool = True, plot_fn = None, **kwargs):
        """
        Train the VAE.

        Args:
            optim (torch.optim): Optimizer.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation data loader.
            n_samples_latent (int, optional): Number of latent samples for ELBO estimation. Default: 25.
            epochs (int, optional): Maximum number of training epochs. Default: 3.
            patience (int, optional): Early stopping patience. Default: 5.
            min_delta (float, optional): Minimum improvement in validation loss. Default: 1e-4.
            reset_history (bool, optional): Whether to reset training history. Default: True.

        Returns:
            VAE: Trained model.
        """
        if reset_history:
            self.reset_history()

        early_stopping = EarlyStopping(patience, min_delta)
        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:

            self.train()
            train_losses = []

            for batch in train_loader:
                optim.zero_grad()
                if self.conditional:
                    loss = -self.elbo(*batch, n_samples_latent)
                else:
                    loss = -self.elbo(batch, n_samples_latent)
                loss.backward()
                optim.step()
                train_losses.append(loss.item())
                torch.cuda.empty_cache()

            avg_train_loss = sum(train_losses) / len(train_losses)
            self.history["Train"].append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_losses = []

                with torch.no_grad():
                    for batch in val_loader:
                        if self.conditional:
                            loss = -self.elbo(*batch, n_samples_latent)
                        else:
                            loss = -self.elbo(batch, n_samples_latent)
                        val_losses.append(loss.item())
                        torch.cuda.empty_cache()

                avg_val_loss = sum(val_losses) / len(val_losses)
                self.history["Val"].append(avg_val_loss)
                pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")

            else:
                pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

            early_stopping.step(avg_val_loss)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
            if plot_fn is not None:
                clear_output(wait=True)
                plot_fn(**kwargs)
        
        self.estimate_prior(train_loader)
        return self
    

    def plot_optim(self,ax=None):
        """
        Plot training and validation loss curves.

        Args:
            ax (matplotlib.Axes, optional): Axis to plot on. If None, creates new figure.

        Returns:
            matplotlib.Axes or (matplotlib.Figure, matplotlib.Axes): Axis or figure/axis tuple.
        """
        state = ax is None
        if state:
            fig,ax = plt.subplots(1,1,figsize=(9,3.5))

        ax.set_title("Objective Minimization")
        ax.set_ylabel("Average Batch Negative ELBO")
        ax.set_xlabel("Iterations")

        ax.plot(
            self.history["Train"],
            label="Training",
            color="tab:red",
            alpha=0.8
        )
        ax.plot(
            self.history["Val"],
            label="Validation",
            color="tab:blue",
            alpha=0.8,
            linestyle="--"
        )

        n_results = len(self.history["Train"])

        ax.set_xticks(
            np.arange(0,n_results),
            np.arange(1,n_results+1)
        )
        ax.grid()
        ax.legend()
        ax.set_axisbelow(True)
        
        if state:
            return ax
        else:
            return fig, ax

    def reset_history(self):
        """
        Reset training history of losses.
        Clears self.history["Train"] and self.history["Val"].
        """
        self.history = {
            "Train":[],
            "Val":[]
        }

    def rsample(self, samples: int, y: torch.Tensor = None, use_estimated_prior:bool = False):
        """
        Generate samples from the VAE's decoder.

        Args:
            samples (int): Number of samples to generate.
            y (Tensor, optional): Conditional labels (if conditional VAE).
            estimate_prior (bool, optional): If True, estimate latent prior from data. Default: False.
            data (DataLoader or Tensor, optional): Data for estimating prior if `estimate_prior=True`.

        Returns:
            Tensor: Generated data samples.
        """
        if use_estimated_prior:
            if self.categorical_conditioned:
                loc = self.__latent_mean[y]
                cov = torch.diagflat(torch.diagonal(self.__latent_cov[y]))
            else:
                loc = self.__latent_mean
                cov = torch.diagflat(torch.diagonal(self.__latent_cov))
        else:    
            loc = torch.zeros(self.latent_dims, device=self.device)
            cov = torch.eye(self.latent_dims, device=self.device)

        z = MVN(loc, cov).rsample((samples,))

        with torch.no_grad():
            if self.conditional:
                if self.one_hot_encode:
                    x_gen = self.decode(z, torch.tensor(y,device=self.device).repeat(samples,))
                else:
                    x_gen = self.decode(z, torch.tensor(y,device=self.device).repeat(samples,1))
            else:
                x_gen = self.decode(z, y)
        return x_gen