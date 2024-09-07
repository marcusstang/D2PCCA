import os
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from pyro.infer import (
    SVI,
    Trace_ELBO,
)
from pyro.optim import (
    Adam,
    ClippedAdam,
)

from D2PCCA import *

def load_data(data_path, sequence_length, test_size):
    df = pd.read_csv(data_path)

    def process_data(df, start_col, end_col):
        # Extract data and convert to numpy array
        data = df.iloc[:, start_col:end_col].astype(float).values

        return data

    # Extract raw data for normalization
    x1_raw = process_data(df, 1, 11)  # Columns 1-11
    x2_raw = process_data(df, 11, 21)  # Columns 11-21
    x3_raw = process_data(df, 21, 31)  # Columns 21-31
    x4_raw = process_data(df, 31, 41)  # Columns 31-41
    x5_raw = process_data(df, 41, 51)  # Columns 41-51

    # Number of sequences in each dataset
    test_start_idx = x1_raw.shape[0] - sequence_length - test_size + 1

    # Split the data into training and testing sets
    train_x1_raw, test_x1_raw = x1_raw[:test_start_idx], x1_raw[test_start_idx:]
    train_x2_raw, test_x2_raw = x2_raw[:test_start_idx], x2_raw[test_start_idx:]
    train_x3_raw, test_x3_raw = x3_raw[:test_start_idx], x3_raw[test_start_idx:]
    train_x4_raw, test_x4_raw = x4_raw[:test_start_idx], x4_raw[test_start_idx:]
    train_x5_raw, test_x5_raw = x5_raw[:test_start_idx], x5_raw[test_start_idx:]

    # Calculate global standard deviation for the entire training set
    global_std = np.concatenate([train_x1_raw, train_x2_raw, train_x3_raw, train_x4_raw, train_x5_raw], axis=0).std(axis=0)

    # Normalize and create sequences for training data
    def create_sequences(data, global_std, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            # Extract each sequence
            sequence = data[i:i + sequence_length]

            # Normalize each segmented sequence using global standard deviation of the training set
            mean = sequence.mean(axis=0)
            normalized_sequence = (sequence - mean) / global_std

            sequences.append(normalized_sequence)

        return torch.tensor(np.array(sequences), dtype=torch.float32)

    # Create sequences for training and testing sets
    train_x1 = create_sequences(train_x1_raw, global_std, sequence_length)
    train_x2 = create_sequences(train_x2_raw, global_std, sequence_length)
    train_x3 = create_sequences(train_x3_raw, global_std, sequence_length)
    train_x4 = create_sequences(train_x4_raw, global_std, sequence_length)
    train_x5 = create_sequences(train_x5_raw, global_std, sequence_length)

    test_x1 = create_sequences(test_x1_raw, global_std, sequence_length)
    test_x2 = create_sequences(test_x2_raw, global_std, sequence_length)
    test_x3 = create_sequences(test_x3_raw, global_std, sequence_length)
    test_x4 = create_sequences(test_x4_raw, global_std, sequence_length)
    test_x5 = create_sequences(test_x5_raw, global_std, sequence_length)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(train_x1, train_x2, train_x3, train_x4, train_x5)
    test_dataset = TensorDataset(test_x1, test_x2, test_x3, test_x4, test_x5)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Check the shapes of the datasets
    print(f"Train dataset size: {len(train_dataset)} sequences")
    print(f"Test dataset size: {len(test_dataset)} sequences")

    return train_loader, test_loader

def save_checkpoint(model, optimizer, epoch, mode, checkpoint_dir="checkpoints"):
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Paths for saving the model and optimizer states
    model_name = f"{mode}_epoch_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    # Save optimizer state using Pyro's save method
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)

    logging.info("Done saving model and optimizer checkpoints to disk.")

def load_checkpoint(model, optimizer, mode, checkpoint_dir="checkpoints", epoch=None):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{mode}")]
    if not model_files:
        logging.info("No checkpoints found. Starting from scratch.")
        return 0

    # Sort checkpoints by epoch number (assuming filenames follow the correct format)
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    if epoch is None:
        # Load the latest checkpoint if no epoch is specified
        model_file = model_files[-1]
        epoch = int(model_file.split('_')[-1].split('.')[0])
    else:
        # Find the checkpoint for the specified epoch
        model_file = f"{mode}_epoch_{epoch}.pth"
        if model_file not in model_files:
            logging.error(
                f"Checkpoint for epoch {epoch} not found. Available epochs: {[int(f.split('_')[-1].split('.')[0]) for f in model_files]}")
            return 0

    model_path = os.path.join(checkpoint_dir, model_file)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_file}")

    # Load model state
    logging.info(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))

    # Load optimizer state using Pyro's load method
    logging.info(f"Loading optimizer states from {optimizer_path}...")
    optimizer.load(optimizer_path)

    logging.info("Done loading model and optimizer states.")
    return epoch

def get_last_saved_epoch(elbo_file_path):
    """Retrieve the last saved epoch from the ELBO pickle file. If the file does not exist, create an empty one."""
    try:
        with open(elbo_file_path, "rb") as f:
            elbo_data = pickle.load(f)
            if not elbo_data:
                return -1  # No epochs saved yet
            # Get the last epoch number
            last_epoch = elbo_data[-1][0]  # Assuming elbo_data is a list of (epoch, elbo) tuples
            return last_epoch
    except (FileNotFoundError, EOFError):
        # File does not exist or is empty, create an empty file
        with open(elbo_file_path, "wb") as f:
            pickle.dump([], f)  # Initialize with an empty list
        return -1  # No epochs saved yet

def main():
    data_path = "/Users/sucra/Desktop/DPCCA/data/stock_prices.csv"
    sequence_length = 30  # T
    test_size = 20  # Use the last 20 sequences for testing
    train_loader, test_loader = load_data(data_path, sequence_length, test_size)

    mode = "kliaf"
    elbo_file_path = f"{mode}_elbo.pkl"  # Path to save the ELBO values

    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cca = D2PCCA(
        x_dims=[10, 10, 10, 10, 10],  # Dimensions for x1 and x2
        emission_dims=[20, 20, 20, 20, 20],  # Hidden dimensions in emission networks
        z_dims=[1, 2, 2, 2, 2, 2],  # Latent state dimensions: shared z, zx, zy
        transition_dims=[5, 10, 10, 10, 10, 10],  # Hidden dimensions in transition networks
        rnn_dim=100,  # RNN hidden dimension
        rnn_layers=1,  # Number of RNN layers
        rnn_dropout_rate=0.1,  # RNN dropout rate
        num_iafs=5,  # Number of IAFs
        iaf_dim=70,  # Hidden dimension for IAFs
    ).to(device)

    # setup optimizer
    adam_params = {
        "lr": 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam = ClippedAdam(adam_params)
    svi = SVI(cca.model, cca.guide, adam, Trace_ELBO(num_particles=10))

    # Load checkpoint if it exists
    start_epoch = load_checkpoint(cca, adam, mode)

    # Get the last saved epoch in the ELBO file
    last_saved_epoch = get_last_saved_epoch(elbo_file_path)

    # Set the epoch to start saving ELBO values from
    save_start_epoch = max(start_epoch, last_saved_epoch + 1)

    # turn on eval mode
    cca.rnn.eval()

    ### EVALUATION I: TEST ELBO ###
    def eval_test_elbo():
        for mini_batch_list in test_loader:
            pass
        test_elbo = -svi.evaluate_loss(mini_batch_list) / (test_size * sequence_length)
        return test_elbo
    #print(-eval_test_elbo())

    ### EVALUATION II: Reconstruction RMSE ###
    def recon_rmse():
        total_rmse = 0.0
        num_samples = 0

        for mini_batch_list in test_loader:
            # No need to extract the first sequence, use all sequences in the mini-batch
            mini_batch_list = [xi.to(cca.device) for xi in mini_batch_list]
            recon, _ = cca(mini_batch_list)

            # Convert mini_batch_list and recon to NumPy arrays for RMSE calculation
            original_data = [xi.cpu().detach().numpy() for xi in mini_batch_list]
            reconstructed_data = [recon[i].cpu().detach().numpy() for i in range(len(recon))]

            # Calculate RMSE for each dimension
            batch_rmse = 0.0
            for d in range(len(original_data)):
                # Compute squared error for each sample, time step, and dimension
                squared_error = (original_data[d] - reconstructed_data[d]) ** 2
                # Sum the squared errors and compute the mean
                batch_rmse += np.mean(squared_error) / sequence_length

            # Compute RMSE for the current batch
            batch_rmse = np.sqrt(batch_rmse)
            total_rmse += batch_rmse
            num_samples += 1

        # Calculate the average RMSE across all batches
        avg_rmse = total_rmse / num_samples
        print(f"Reconstruction RMSE: {avg_rmse:.4f}")
        return avg_rmse
    recon_rmse()

    ### Visualization I: Reconstruction ###
    def recon_fig():
        dim_idx = 0

        '''
        # Get a single mini-batch from the train_loader
        seq_idx = 400
        i = 4
        for mini_batch_list in train_loader:
            if i == seq_idx:
                # Extract the first sequence from the mini_batch_list for all xi
                mini_batch_list = [xi[:1, :, :] for xi in mini_batch_list]
                break  # Only use the first mini-batch for reconstruction
            i += 1

        '''
        for mini_batch_list in test_loader:
            mini_batch_list = [xi[:1, :, :] for xi in mini_batch_list]
            break  # Only use the first mini-batch for reconstruction

        # Perform reconstruction using the forward method of the D2PCCA model
        recon, scales = cca(mini_batch_list)

        # Select the original and reconstructed sequences for x1
        original_x1 = mini_batch_list[0].cpu().detach().numpy()  # Convert to NumPy array for plotting
        recon_x1 = recon[0].cpu().detach().numpy()  # Reconstructed x1 sequence
        scale_x1 = scales[0].cpu().detach().numpy()  # Scale (standard deviation) for x1 sequence

        # Extract the first feature (dimension) for the first sample in the batch
        original_x1_first_feature = original_x1[0, :, dim_idx]  # Shape: [lengths]
        recon_x1_first_feature = recon_x1[0, :, dim_idx]  # Shape: [lengths]
        scale_x1_first_feature = scale_x1[0, :, dim_idx]

        # Calculate 95% CI for the reconstructed sequence
        lower_bound = recon_x1_first_feature - 1.96 * scale_x1_first_feature
        upper_bound = recon_x1_first_feature + 1.96 * scale_x1_first_feature

        # Plot the original and reconstructed sequences for the first feature of x1
        plt.figure(figsize=(10, 6))
        plt.plot(original_x1_first_feature, label='Original x1[0,:,0]', linestyle='-', marker='o')
        plt.plot(recon_x1_first_feature, label='Reconstructed x1[0,:,0]', linestyle='--', marker='x')
        plt.fill_between(range(len(recon_x1_first_feature)), lower_bound, upper_bound, color='gray', alpha=0.3,
                         label='95% CI')
        plt.title('Original vs Reconstructed Sequence for x1[0,:,0]')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    #recon_fig()

    ### Visualization II: Training & Test ELBO ###
def plot_elbo(mode):
    elbo_file_path = f"{mode}_elbo.pkl"
    # Load ELBO data from the pickle file
    try:
        with open(elbo_file_path, "rb") as f:
            elbo_data = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {elbo_file_path}")
        return
    except EOFError:
        print(f"File is empty or corrupted: {elbo_file_path}")
        return

    # Separate the data into epochs, train_elbo, and test_elbo
    epochs = [entry[0] for entry in elbo_data]
    train_elbo = [entry[1] for entry in elbo_data]
    test_elbo = [entry[2] for entry in elbo_data]

    # Plot the training and test ELBO values
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_elbo, label='Training ELBO', linestyle='-', marker='o')
    plt.plot(epochs, test_elbo, label='Test ELBO', linestyle='--', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.title('Training and Test ELBO over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    main()
    #plot_elbo("kliaf")

