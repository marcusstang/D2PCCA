import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

import pyro
import pyro.distributions as dist


class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


class LinearTransition(nn.Linear):
    def __init__(self, latent_dims):
        # latent_dims: a tuple containing the dimensions of (zs_t, z1_t, z2_t, ...)
        self.latent_dims = latent_dims
        self.shared_dim = latent_dims[0]
        self.individual_dims = latent_dims[1:]
        self.total_latent_dim = sum(latent_dims)

        super(LinearTransition, self).__init__(self.total_latent_dim, self.total_latent_dim)

        # Construct the mask
        mask = np.zeros((self.total_latent_dim, self.total_latent_dim))
        mask[:self.shared_dim, :self.shared_dim] = 1  # A_0 in the top-left block
        start_idx = self.shared_dim
        for i, dim in enumerate(self.individual_dims):
            mask[start_idx:start_idx + dim, start_idx:start_idx + dim] = 1
            start_idx += dim

        mask = torch.from_numpy(mask).float()

        # Register the mask buffer
        self.register_buffer('mask', mask)

    def forward(self, prev_state):
        # Apply the mask to the weight matrix
        self.weight.data *= self.mask
        # Perform the linear transformation with masked weights
        z_t = super(LinearTransition, self).forward(prev_state)
        return z_t


class LinearEmission(nn.Linear):
    def __init__(self, obs_dims, latent_dims):
        self.latent_dims = latent_dims
        self.shared_dim = latent_dims[0]
        self.individual_dims = latent_dims[1:]
        self.total_latent_dim = sum(latent_dims)
        self.obs_dims = obs_dims
        self.total_obs_dim = sum(obs_dims)

        super(LinearEmission, self).__init__(self.total_latent_dim, self.total_obs_dim)

        # Construct the mask
        mask = np.zeros((self.total_obs_dim, self.total_latent_dim))
        start_i = 0
        for i, dim_x in enumerate(self.obs_dims):
            mask[start_i:start_i + dim_x, 0:self.shared_dim] = 1
            start_j = self.shared_dim + sum(self.individual_dims[:i])
            end_j = start_j + self.individual_dims[i]
            mask[start_i:start_i + dim_x, start_j:end_j] = 1
            start_i += dim_x
        mask = torch.from_numpy(mask).float()

        # Register the mask buffer
        self.register_buffer('mask', mask)

    def forward(self, z_t):
        # Apply the mask to the weight matrix
        self.weight.data *= self.mask
        # Perform the linear transformation with masked weights
        x_t = super(LinearEmission, self).forward(z_t)
        return x_t


# DPCCA class
class DPCCA(nn.Module):
    def __init__(self, obs_dims, latent_dims, rnn_dim=10):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")

        self.obs_dims = obs_dims  # List of dimensions for each observed variable [dim(x1), dim(x2)]
        self.latent_dims = latent_dims  # List of dimensions for each latent state [dim(z0), dim(z1), dim(z2)]
        self.shared_dim = latent_dims[0]  # Dimension of the shared latent state (z0)
        self.individual_dims = latent_dims[1:]  # Dimensions of the individual latent states (z1, z2)
        self.total_latent_dim = sum(latent_dims)  # Total dimension of the latent state vector z_t
        self.total_obs_dim = sum(obs_dims)  # Total dimension of the observed state vector x_t
        self.rnn_dim = rnn_dim

        self.state_trans = LinearTransition(latent_dims)
        self.emission = LinearEmission(obs_dims, latent_dims)

        # Initialize submatrices for transitions and emissions
        self.V_shared = nn.Parameter(torch.eye(self.shared_dim))  # V_0
        self.V_individuals = nn.ParameterList(
            [nn.Parameter(torch.eye(dim)) for dim in self.individual_dims])  # V_1, V_2
        # Initialize submatrices for emissions
        self.log_sigmas = nn.Parameter(torch.zeros(len(obs_dims)))  # Log variances for emissions

        # Learnable initial states
        self.z_0 = nn.Parameter(torch.zeros(self.total_latent_dim))  # Combined initial latent state
        self.z_q_0 = nn.Parameter(torch.zeros(self.total_latent_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        # Inference network
        self.rnn = nn.RNN(
            input_size=self.total_obs_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
        )
        self.combiner = Combiner(self.total_latent_dim, rnn_dim)

    def model(self, mini_batch_list):
        pyro.module("dpcca", self)
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size

        # Construct the full covariance matrix V_hat from submatrices
        V_hat = torch.zeros(self.total_latent_dim, self.total_latent_dim, device=self.device)
        V_hat[:self.shared_dim, :self.shared_dim] = self.V_shared @ self.V_shared.T  # V_0 in the top-left block
        start_idx = self.shared_dim
        for i, V_ind in enumerate(self.V_individuals):
            V_hat[start_idx:start_idx + self.individual_dims[i],
            start_idx:start_idx + self.individual_dims[i]] = V_ind @ V_ind.T
            start_idx += self.individual_dims[i]

        # noise of the emission model
        Sigma = torch.zeros(self.total_obs_dim, self.total_obs_dim, device=self.device)
        start_idx = 0
        for i in range(len(self.log_sigmas)):
            Sigma[start_idx:start_idx + self.obs_dims[i], start_idx:start_idx + self.obs_dims[i]] = torch.exp(
                self.log_sigmas[i]) * torch.eye(self.obs_dims[i], device=self.device)
            start_idx += self.obs_dims[i]

        # Initialize the previous latent state
        z_prev = self.z_0.expand(batch_size, self.total_latent_dim)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # State transition
                z_loc = self.state_trans(z_prev)
                z_cov = V_hat.expand(batch_size, -1, -1)
                z_t = pyro.sample(f"z_{t}", dist.MultivariateNormal(z_loc, z_cov))

                # Emission model
                x_loc = self.emission(z_t)
                x_cov = Sigma.expand(batch_size, -1, -1)
                pyro.sample(f"x_{t}", dist.MultivariateNormal(x_loc, x_cov),
                            obs=torch.cat(mini_batch_list, dim=-1)[:, t - 1, :])

                # Update for the next time step
                z_prev = z_t

    def guide(self, mini_batch_list):
        pyro.module("dpcca", self)
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]
        # Use the first mini-batch to determine sequence length and batch size
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size
        # Combine all mini-batches along the feature dimension for RNN input
        mini_batch_combined = torch.cat(mini_batch_list, dim=-1)
        # Expand h_0 to fit batch size
        h_0_contig = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        # Reverse the combined mini-batch
        mini_batch_reversed = torch.flip(mini_batch_combined, dims=[1])

        # Pass through the RNN
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # ST-R: q(z_t | z_{t-1}, x_{t:T}, y_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                z_t = pyro.sample(f"z_{t}", z_dist)
                # Update time step
                z_prev = z_t

    def forward(self, mini_batch_list):
        # Ensure mini_batch_list is on the correct device
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]

        # Use the first mini-batch to determine sequence length and batch size
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size
        # Combine all mini-batches along the feature dimension for RNN input
        mini_batch_combined = torch.cat(mini_batch_list, dim=-1)
        # Expand h_0 to fit batch size
        h_0_contig = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        # Reverse the combined mini-batch
        mini_batch_reversed = torch.flip(mini_batch_combined, dims=[1])

        # Pass through the RNN
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        # Calculate scales (standard deviations)
        x_scale = torch.exp(0.5 * self.log_sigmas)  # Convert log variances to standard deviations

        reconstructions = []
        scales = []

        # Generate latent states using the guide
        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # Infer the latent state z_t using the guide
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                z_all_t = z_loc

                # Reconstruct the observations using the emission model
                x_loc = self.emission(z_all_t)
                ctr = 0
                for i, dim in enumerate(self.obs_dims):
                    reconstructions.append(x_loc[:, ctr:ctr+dim])
                    scales.append(x_scale[i].expand(batch_size, dim))
                    ctr += dim

                # Update for the next time step
                z_prev = z_all_t

        # Group reconstructions by each observation x1, x2, ..., x5
        grouped_reconstructions = [
            torch.stack([reconstructions[t] for t in range(i, len(reconstructions), len(self.obs_dims))], dim=1)
            for i in range(len(self.obs_dims))]
        grouped_scales = [
            torch.stack([scales[t] for t in range(i, len(scales), len(self.obs_dims))], dim=1)
            for i in range(len(self.obs_dims))
        ]

        return grouped_reconstructions, grouped_scales


def train(svi, train_loader, num_sectors=2):
    epoch_nll = 0.0
    for which_mini_batch, mini_batch_list in enumerate(train_loader):
        mini_batch_list = mini_batch_list[:num_sectors]
        epoch_nll += svi.step(mini_batch_list)

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_nll / normalizer_train
    return total_epoch_loss_train

