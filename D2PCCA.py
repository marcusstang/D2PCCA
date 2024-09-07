import torch
import torch.nn.functional as F
import torch.nn as nn

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive

# p(x_t | zs_t, zu_t)
class Emitter(nn.Module):
    def __init__(self, x_dim, zs_dim, zu_dim, emission_dim):
        # x_dim:        dimensions of the output
        # zs_dim:       dimensions of the shared latent state
        # zu_dim:       dimensions of the unique latent state
        # emission_dim: dimensions of the hidden layer of the emission network
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(zs_dim + zu_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, x_dim)
        self.relu = nn.ReLU()

    def forward(self, zs_t, zc_t):
        z_combined = torch.cat((zs_t, zc_t), dim=-1)
        h1 = self.relu(self.lin_z_to_hidden(z_combined))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        x_loc = self.lin_hidden_to_loc(h2)
        x_scale = torch.exp(self.lin_hidden_to_scale(h2))
        return x_loc, x_scale

# p(z_t | z_{t-1})
# The latent state z_t is tructured as (zs_t, z1_t, z2_t, ..., zd_t), where zs_t
# is the latent state shared by each observed variable, and zi_t is the latent
# state unique to the i-th observed variable, for i = 1,...,d.
class GatedTransition(nn.Module):
    def __init__(self, z_dims, transition_dims):
        # z_dims:           a variable-length tuple containing the dimensions of
        #                   zs_t, z1_t, z2_t, ..., and zd_t, respectively.
        # transition_dims:  dimensions of the hidden layers of the transition
        #                   network. It may take a tuple of the same shape as the
        #                   z_dims, or a single integer if the hidden dimensions
        #                   are of the same size.
        super().__init__()
        self.d = len(z_dims)
        self.z_dims = z_dims

        if isinstance(transition_dims, int):
            transition_dims = [transition_dims] * self.d
        else:
            assert len(transition_dims) == self.d

        self.shared_transition = self._build_transition_layers(z_dims[0], transition_dims[0])

        # Create transition layers for each observed variable's latent state
        self.individual_transitions = nn.ModuleList([
            self._build_transition_layers(z_dims[i], transition_dims[i]) for i in range(1, self.d)
        ])

    def forward(self, z_t):
        # Assert that the input dimension matches the expected sum of z_dims
        assert z_t.shape[-1] == sum(self.z_dims), (
            f"Expected z_t to have last dimension {sum(self.z_dims)}, "
            f"but got {z_t.shape[-1]} instead."
        )

        # Extract the shared latent state (zs_t)
        zs_t = z_t[:, :self.z_dims[0]]
        shared_loc, shared_scale = self._compute_transition(zs_t, self.shared_transition)

        # For each individual latent state (z1_t, z2_t, ..., zd_t)
        individual_locs = []
        individual_scales = []
        start_idx = self.z_dims[0]  # Start index after the shared latent state
        for i in range(1, self.d):
            zi_t = z_t[:, start_idx:start_idx + self.z_dims[i]]
            zi_loc, zi_scale = self._compute_transition(zi_t, self.individual_transitions[i - 1])
            individual_locs.append(zi_loc)
            individual_scales.append(zi_scale)
            start_idx += self.z_dims[i]  # Update the start index for the next individual latent state

        # Concatenate the locations and scales for zs_t and each zi_t
        cca_loc = torch.cat([shared_loc] + individual_locs, dim=-1)
        cca_scale = torch.cat([shared_scale] + individual_scales, dim=-1)

        return cca_loc, cca_scale


    def _build_transition_layers(self, z_dim, transition_dim):
        layers = nn.ModuleDict({
            'lin_gate_z_to_hidden': nn.Linear(z_dim, transition_dim),
            'lin_gate_hidden_to_z': nn.Linear(transition_dim, z_dim),
            'lin_proposed_mean_z_to_hidden': nn.Linear(z_dim, transition_dim),
            'lin_proposed_mean_hidden_to_z': nn.Linear(transition_dim, z_dim),
            'lin_sig': nn.Linear(z_dim, z_dim),
            'lin_z_to_loc': nn.Linear(z_dim, z_dim),
        })

        # Initialize `lin_z_to_loc` to be an identity mapping initially
        nn.init.eye_(layers['lin_z_to_loc'].weight)
        nn.init.zeros_(layers['lin_z_to_loc'].bias)

        return layers

    def _compute_transition(self, z_t, layers):
        # g_t: gating units
        _gate = F.relu(layers['lin_gate_z_to_hidden'](z_t))
        gate = torch.sigmoid(layers['lin_gate_hidden_to_z'](_gate))

        # h_t: proposed mean
        _proposed_mean = F.relu(layers['lin_proposed_mean_z_to_hidden'](z_t))
        proposed_mean = layers['lin_proposed_mean_hidden_to_z'](_proposed_mean)

        # loc, scale
        z_loc = (1 - gate) * layers['lin_z_to_loc'](z_t) + gate * proposed_mean
        z_scale = F.softplus(layers['lin_sig'](F.relu(proposed_mean)))

        return z_loc, z_scale

# q(z_t|z_{t-1}, h_t^r)
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

class D2PCCA(nn.Module):
    def __init__(self, x_dims, emission_dims, z_dims, transition_dims, rnn_dim,
                 rnn_layers=1, rnn_dropout_rate=0.0, num_iafs=0, iaf_dim=50):
        # x_dims and emission_dims are defined similarly to z_dims and transition_dims
        # rnn_dim: hidden size of RNN used in the inference process.
        # rnn_layers: number of layers in the RNN.
        # rnn_dropout_rate: dropout rate used in the RNN.
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")

        # Validate input dimensions
        assert len(x_dims) == len(emission_dims), "x_dims and emission_dims must have the same length."
        assert len(z_dims) == len(transition_dims), "z_dims and transition_dims must have the same length."

        self.d = len(z_dims)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.z_shared_dim = z_dims[0]  # The first dimension is for the shared latent state (z)
        self.z_individual_dims = z_dims[1:]  # Remaining dimensions are for individual latent states (zx, zy, etc.)

        # Initialize emitters for each output variable
        self.emitters = nn.ModuleList([
            Emitter(x_dims[i], z_dims[0], z_dims[i + 1], emission_dims[i]) for i in range(self.d - 1)
        ])

        # Initialize the transition model
        self.trans = GatedTransition(z_dims, transition_dims)

        # Initialize the combiner for inference
        self.combiner = Combiner(sum(z_dims), rnn_dim)

        # RNN configuration
        self.rnn = nn.RNN(
            input_size=sum(x_dims),
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=rnn_layers,
            dropout=rnn_dropout_rate if rnn_layers > 1 else 0.0,
        )

        # Normalizing flows (IAF)
        self.iafs = [
            affine_autoregressive(sum(z_dims), hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # p(z_0) initialization
        self.z_0 = nn.Parameter(torch.zeros(sum(z_dims)))
        # q(z_0)
        self.z_q_0 = nn.Parameter(torch.zeros(sum(z_dims)))
        # Initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    # p(z_t, x_t|z_{t-1}, x_{t-1})
    def model(self, mini_batch_list, annealing_factor=1.0):
        pyro.module("D2PCCA", self)
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size

        # Ensure all mini-batches have the same shape
        for mini_batch in mini_batch_list:
            assert mini_batch.size(1) == T_max and mini_batch.size(0) == batch_size, \
                "All mini-batches must have the same sequence length and batch size."

        # p(z_0)
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))  # Replicate z_0 for the batch

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # p(z_t | z_{t-1})
                z_all_loc, z_all_scale = self.trans(z_prev)
                with poutine.scale(scale=annealing_factor):
                    z_all_t = pyro.sample(f"z_{t}", dist.Normal(z_all_loc, z_all_scale).to_event(1))

                # Extract individual latent states (z, zx, zy, etc.)
                z_t = z_all_t[:, :self.z_shared_dim]
                individual_latents = []
                start_idx = self.z_shared_dim
                for i in range(self.d - 1):
                    individual_latents.append(z_all_t[:, start_idx:start_idx + self.z_individual_dims[i]])
                    start_idx += self.z_individual_dims[i]

                # Compute emissions for each output variable
                for i, emitter in enumerate(self.emitters):
                    x_loc, x_scale = emitter(z_t, individual_latents[i])
                    pyro.sample(f"obs_x{i + 1}_{t}", dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch_list[i][:, t - 1, :])

                # Update time step
                z_prev = z_all_t


    def guide(self, mini_batch_list, annealing_factor=1.0):
        pyro.module("D2PCCA", self)
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
                z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs) if self.iafs else dist.Normal(z_loc, z_scale).to_event(1)
                with poutine.scale(scale=annealing_factor):
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

        reconstructions = []
        scales = []

        # Generate latent states using the guide
        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                if self.iafs:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    z_all_t = pyro.sample(f"z_{t}", z_dist)
                else:
                    z_all_t = z_loc

                # Extract individual latent states (z, zx, zy, etc.)
                z_t = z_all_t[:, :self.z_shared_dim]
                individual_latents = []
                start_idx = self.z_shared_dim
                for i in range(self.d - 1):
                    individual_latents.append(z_all_t[:, start_idx:start_idx + self.z_individual_dims[i]])
                    start_idx += self.z_individual_dims[i]

                # Compute emissions for each output variable
                for i, emitter in enumerate(self.emitters):
                    x_loc, x_scale = emitter(z_t, individual_latents[i])
                    reconstructions.append(x_loc)
                    scales.append(x_scale)

                # Update time step
                z_prev = z_all_t

        #print(x_loc.shape)

        # Group reconstructions by each observation x1, x2, ..., x5
        grouped_reconstructions = [
            torch.stack([reconstructions[t] for t in range(i, len(reconstructions), self.d - 1)], dim=1)
            for i in range(self.d - 1)]
        grouped_scales = [
            torch.stack([scales[t] for t in range(i, len(scales), self.d - 1)], dim=1)
            for i in range(self.d - 1)
        ]

        return grouped_reconstructions, grouped_scales


def train_KL_annealing(svi, train_loader, epoch, annealing_epochs, minimum_annealing_factor, num_sectors=2):
    batch_size = train_loader.batch_size
    N_mini_batches = len(train_loader)
    epoch_nll = 0.0
    for which_mini_batch, mini_batch_list in enumerate(train_loader):
        mini_batch_list = mini_batch_list[:num_sectors]
        if annealing_epochs > 0 and epoch < annealing_epochs:
            annealing_factor = minimum_annealing_factor + (1.0 - minimum_annealing_factor) * (
                float(which_mini_batch + epoch * N_mini_batches + 1)
                / float(annealing_epochs * N_mini_batches)
            )
        else:
            annealing_factor = 1.0

        # Pass all observed variables (mini-batches) to SVI
        epoch_nll += svi.step(mini_batch_list, annealing_factor)

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_nll / normalizer_train
    return total_epoch_loss_train


def train(svi, train_loader, num_sectors=2):
    batch_size = train_loader.batch_size
    N_mini_batches = len(train_loader)
    epoch_nll = 0.0
    for which_mini_batch, mini_batch_list in enumerate(train_loader):
        mini_batch_list = mini_batch_list[:num_sectors]
        epoch_nll += svi.step(mini_batch_list)

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_nll / normalizer_train
    return total_epoch_loss_train