import mne
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as plx
from mne.datasets import misc
import mne_connectivity
import pyxdf
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

channels_to_use = [# prefrontal
    'FC3',
    'FCz',
    'FC4',
    # central and temporal
    'C5',
    'C3',
    'C1',
    'Cz',
    'C2',
    'C4',
    'C6',
    'CP3',
    'CP1',
    'CPz',
    'CP2',
    'CP4',
    'Pz']

def build_mne_object(fname, eeg_stream_name='Gwennie-24', stimulus_stream = 'stimulus_stream'):
    streams, header = pyxdf.load_xdf(fname)

    # Find the index of the stimulus and EEG streams
    eeg_index = []
    for stream in range(len(streams)):
        if streams[stream]["info"]["name"][0] == eeg_stream_name:
            eeg_index.append(stream)
    
    # The EEG channels are assumed to be constant across streams 
    # because this is built into the DSI-24 system
    eeg_index1 = eeg_index[0]
    ch_names = []
    for i in range(0, len(streams[eeg_index1]["info"]["desc"][0]["channels"][0]["channel"])):
        ch_names.append(streams[eeg_index1]["info"]["desc"][0]["channels"][0]["channel"][i]["label"][0])
    
    # Create the info object
    samp_frq = float(streams[eeg_index1]["info"]["nominal_srate"][0])
    ch_types = ['eeg'] * len(ch_names)

    # Find the stimulus stream in streams
    stimulus_stream = None
    for stream in range(len(streams)):
        if streams[stream]["info"]["name"][0] == "stimulus_stream":  # Match name
            stimulus_stream = streams[stream]
            break

    if stimulus_stream is None:
        raise ValueError("No 'stimulus_stream' found in the dataset.")


    # Extract stimulus timestamps and event markers
    first_timestamp =float(stimulus_stream["footer"]["info"]["first_timestamp"][0])

    event_timestamps = stimulus_stream["time_stamps"] 
    eeg_timestamps = streams[eeg_index1]["time_stamps"]
    event_index = np.searchsorted(eeg_timestamps, event_timestamps)

    event_dict = stimulus_stream["time_series"].flatten()  # Convert to 1D array

    # format the events array to correspond to what MNE expects
    events = np.column_stack([
        (event_index).astype(int),
        np.zeros(len(event_timestamps), dtype=int),
        event_dict
    ])

    info = mne.create_info(ch_names, sfreq = samp_frq, ch_types= ch_types, verbose=None)

    # Create the raw object 
    # Here we assume that there is only one EEG stream    
    eeg_data = streams[eeg_index1]["time_series"].T
    # # uV -> V
    eeg_data *= 1e-6  
    raw = mne.io.RawArray(eeg_data, info, verbose=None)

    fs = raw.info['sfreq']
    print(f'Frequency of Sampling: {fs} Hz')
    # Length in seconds
    print(f'Duration: {len(raw) / fs} seconds')

    return raw, events, samp_frq

def extract_clean_events(trig, fs):
    trig = trig.squeeze()
    clean_trig = np.zeros_like(trig)

    prev = 0
    for i in range(1, len(trig)):
        curr = trig[i]
        # Detect rising edge (start of a new block)
        if prev == 0 and curr in (1, -1):
            # Compute time point 2 seconds after onset
            time_idx = i + int(2 * fs)
            if time_idx < len(trig):
                clean_trig[time_idx] = 2 if curr == 1 else -2
                clean_trig[i] = 1
        prev = curr

    return clean_trig

def set_1020_montage(raw):
    """
    Set the 10-20 montage for the EEG data.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    """
    # Set the montage to 10-20 system
    sample_1020 = raw.copy().pick_channels(channels_to_use)
    assert len(channels_to_use) == len(sample_1020.ch_names)
    ch_map = {ch.lower(): ch for ch in sample_1020.ch_names}
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    len(ten_twenty_montage.ch_names)
    ten_twenty_montage.ch_names = [ch_map[ch.lower()] if ch.lower() in ch_map else ch 
                                for ch in ten_twenty_montage.ch_names]
    sample_1020.set_montage(ten_twenty_montage)
    return sample_1020



def extract_power_matrix(tfr=None, freq_bins=None):
    max_t = tfr.times[-1]
    time_bins = [(start, min(start + 0.3, max_t)) for start in np.arange(0, max_t, 0.3)]
    n_channels = len(tfr.ch_names)
    mat = np.zeros((n_channels, len(freq_bins), len(time_bins)))
    labels = []

    for i_f, (fmin, fmax) in enumerate(freq_bins):
        for i_t, (tmin, tmax) in enumerate(time_bins):
            tf_crop = tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
            mean_power = tf_crop.data.mean(axis=(1, 2))  # shape: (n_channels,)
            mat[:, i_f, i_t] = mean_power

    # Generate labels
    for ch in tfr.ch_names:
        for i_f, (fmin, fmax) in enumerate(freq_bins):
            for i_t, (tmin, tmax) in enumerate(time_bins):
                labels.append(('power', ch, f'{fmin:.1f}-{fmax:.1f}Hz_{tmin:.1f}-{tmax:.1f}s'))

    return mat, labels



def extract_power_features(epoch, freqs, freq_bins, channels):
    """
    Extracts power features from a single epoch.

    Parameters:
    - epoch: instance of mne.Epochs containing a single epoch
    - freqs: array of frequencies to analyze
    - freq_bins: list of (fmin, fmax) frequency bins
    - channels: list of channel names to use

    Returns:
    - features: 1D numpy array of power features
    - labels: list of tuples (feature_type, channel, detail)
    """
    n_cycles = freqs / 2.

    # Compute TFR for this single epoch
    tfr = mne.time_frequency.tfr_multitaper(
        epoch,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        picks='eeg'
    )

    max_t = tfr.times[-1]
    time_bins = [(start, min(start + 0.3, max_t)) for start in np.arange(0, max_t, 0.3)]
    n_channels = len(tfr.ch_names)

    features = []
    labels = []

    for ch_idx, ch_name in enumerate(tfr.ch_names):
        for fmin, fmax in freq_bins:
            for tmin, tmax in time_bins:
                tf_crop = tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
                mean_power = tf_crop.data[ch_idx].mean()  # scalar
                features.append(mean_power)
                labels.append(('power', ch_name, f'{fmin:.1f}-{fmax:.1f}Hz_{tmin:.1f}-{tmax:.1f}s'))

    return np.array(features), labels





def create_power_df(epoch_dict, freqs, freq_bins = None):
    n_cycles = freqs / 2.  # typical choice for multitaper
    power_dict = {}
    for epoch_type, epoch in epoch_dict.items():
        #power = tfr_multitaper(epoch, freqs=freqs, n_cycles=n_cycles, return_itc=False, picks='eeg')
        power = epoch.compute_tfr( method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,         
            picks='eeg',
            )
        matrix, time_bins = extract_power_matrix(tfr = power, freq_bins = freq_bins)
       
       # Append the list of channel-wise power values
        power_dict[epoch_type] = [matrix[i] for i in range(len(channels_to_use))]

    #Create the DataFrame 
    power_df = pd.DataFrame(power_dict, index=channels_to_use)
    
    return power_df


def create_corr_df(epoch_dict, plot=False):
    task_types = list(epoch_dict.keys())
    channel_pairs = list(combinations(channels_to_use, 2))

    # Initialize DataFrame to store correlation values
    fc_df = pd.DataFrame(index=[f"{ch1}-{ch2}" for ch1, ch2 in channel_pairs],
                        columns=task_types)


    # Loop over epoch types and get data
    for epoch_type, epochs in epoch_dict.items():
        epochs_data = epochs.get_data()  # shape: (n_channels, n_times) -> because we take average within this epoch
        print(f"Epoch Type: {epoch_type}, Shape: {epochs_data.shape}")
        ch_names = epochs.info['ch_names']
        epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        n_channels = epochs_data.shape[0]

        # Initialize correlation matrix
        corr_matrix = np.zeros((n_channels, n_channels))

        for ch1, ch2 in channel_pairs:
            ch1_idx = ch_names.index(ch1)
            ch2_idx = ch_names.index(ch2)

            # Get the data for the two channels across all epochs and time points
            #print(f"Computing correlation for {ch1} and {ch2}")
            data_ch1 = epochs_data[ch1_idx, :]
            data_ch2 = epochs_data[ch2_idx, :]

            # Compute Pearson correlation for the two channels
            fc_df.loc[f"{ch1}-{ch2}", epoch_type] = pearsonr(data_ch1.flatten(), data_ch2.flatten())[0]  # correlation coefficient
            corr_matrix[ch1_idx, ch2_idx] = fc_df.loc[f"{ch1}-{ch2}", epoch_type]
            corr_matrix[ch2_idx, ch1_idx] = fc_df.loc[f"{ch1}-{ch2}", epoch_type]

        # Plot the matrix
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, xticklabels=ch_names, yticklabels=ch_names, 
                        cmap='RdBu_r', center=0, annot=False, square=True)
            plt.title(f"Functional Connectivity {epoch_type}")
            plt.tight_layout()
            plt.show()

    return fc_df, corr_matrix

def build_mne_object_from_data(y,channels_to_use, sfreq):
    """
    Build an MNE Raw object from EEG data and corresponding event information.
    
    Parameters:
    - y: EEG data (shape should be [n_channels, n_samples])
    - trig: Event trigger information (shape should be [n_events, 1])
    - channels_to_use: List of EEG channel names
    - sfreq: Sampling frequency
    
    Returns:
    - raw: MNE Raw object containing EEG data
    - events: MNE events array formatted for MNE
    - samp_frq: Sampling frequency
    """
    # Create the info object
    ch_names = channels_to_use
    
    
    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=None)
    
    # Create the raw object using the provided EEG data
    eeg_data = np.array(y.T)  # Transpose if needed to match MNE's [n_channels, n_samples] shape
    raw = mne.io.RawArray(eeg_data, info, verbose=None)
    
    
    # Display information about the raw data
    fs = raw.info['sfreq']
    print(f'Frequency of Sampling: {fs} Hz')
    print(f'Duration: {len(raw) / fs} seconds')

    return raw, sfreq


def create_temp_df(epoch_dict, sampling_freq):
    sfreq = sampling_freq
    segment_duration = 0.3 
    seg_len = int(segment_duration * sfreq) #number of samples in each segment

    mstd_df = pd.DataFrame(index=channels_to_use, columns=epoch_dict.keys())

    for epoch_type, epochs in epoch_dict.items():
        evoked = epochs.copy().pick(channels_to_use)
        data = evoked.data 
        ch_names = evoked.ch_names
        n_channels, n_times = data.shape
        n_segments = n_times // seg_len #15

        for ch_idx, ch_name in enumerate(ch_names):
            mean_list = []
            std_list = []

            for s in range(n_segments):
                s_start = s * seg_len
                s_end = s_start + seg_len
                seg = data[ch_idx, s_start:s_end] 
                mean_list.append(np.mean(seg))
                std_list.append(np.std(seg))

            mstd_df.at[ch_name, epoch_type] = [mean_list, std_list] # loc didn't work that's why I used at

    return mstd_df

def extract_temp_features(epoch_data, sfreq, channels):
    """
    Extracts temporal features (mean and std of segments) from a single epoch.

    Parameters:
    - epoch_data: np.ndarray of shape (n_channels, n_times)
    - sfreq: sampling frequency
    - channels: list of channel names in the same order as in epoch_data

    Returns:
    - features: 1D numpy array of features
    - labels: list of tuples (feature_type, channel, detail)
    """

    segment_duration = 0.3  # seconds
    seg_len = int(segment_duration * sfreq)  # samples per segment
    n_channels, n_times = epoch_data.shape
    n_segments = n_times // seg_len

    features = []
    labels = []

    for ch_idx, ch_name in enumerate(channels):
        ch_data = epoch_data[ch_idx]

        for s in range(n_segments):
            s_start = s * seg_len
            s_end = s_start + seg_len
            seg = ch_data[s_start:s_end]

            mean_val = np.mean(seg)
            std_val = np.std(seg)

            features.append(mean_val)
            labels.append(('temp', ch_name, f'mean_seg_{s}'))

            features.append(std_val)
            labels.append(('temp', ch_name, f'std_seg_{s}'))

    return np.array(features), labels


def create_final_df(power_df=None, fc_df=None, mstd_df=None):
    dataframes = []

    if power_df is not None:
        freq_df = power_df.copy()
        freq_df.index = pd.MultiIndex.from_product([['freq_features'], freq_df.index])
        dataframes.append(freq_df)

    if fc_df is not None:
        corr_df = fc_df.copy()
        corr_df.index = pd.MultiIndex.from_product([['corr_features'], corr_df.index])
        dataframes.append(corr_df)

    if mstd_df is not None:
        mstd_df = mstd_df.copy()
        mstd_df.index = pd.MultiIndex.from_product([['mstd_features'], mstd_df.index])
        dataframes.append(mstd_df)

    # Concatenate all non-None DataFrames
    if dataframes:
        combined_df = pd.concat(dataframes)
    else:
        combined_df = pd.DataFrame()  # Return empty if all inputs are None

    return combined_df

import re
import pandas as pd

def map_labels_to_segments(df):
    time_power_feat = ['0.0-0.1s', '0.1-0.2s', '0.2-0.3s', '0.3-0.4s', '0.4-0.5s', '0.5-0.6s', '0.6-0.7s', '0.7-0.8s', '0.8-0.9s', '0.9-1.0s', '1.0-1.1s', '1.1-1.2s', '1.2-1.3s', '1.3-1.4s', '1.4-1.5s']
    segmentation = ['seg_0', 'seg_1', 'seg_2', 'seg_3', 'seg_4', 'seg_5', 'seg_6', 'seg_7', 'seg_8', 'seg_9', 'seg_10', 'seg_11', 'seg_12', 'seg_13', 'seg_14']
    
    mapping = dict(zip(time_power_feat, segmentation))
    new_columns = []

    for feat_type, channel, detail in df.columns:
        if feat_type == "power":
            # Find the time portion like '0.0-0.1s' at the end
            match = re.search(r'(\d+\.\d+-\d+\.\d+s)$', detail)
            if match:
                time_str = match.group(1)
                if time_str in mapping:
                    seg_label = mapping[time_str]
                    # Replace the time portion with the corresponding seg_X
                    new_detail = re.sub(r'(\d+\.\d+-\d+\.\d+s)$', seg_label, detail)
                else:
                    new_detail = detail  # fallback in case of unexpected format
            else:
                new_detail = detail
        else:
            new_detail = detail  # leave temporal features unchanged

        new_columns.append((feat_type, channel, new_detail))

    df.columns = pd.MultiIndex.from_tuples(new_columns, names=df.columns.names)
    return df

import numpy as np
import pandas as pd
import re
import torch.nn as nn

def reshape_features_by_segment(df):
    features_df = df.drop(columns="label")
    labels = df["label"].values
    n_epochs = len(df)

    # Extract segment indices
    seg_pattern = re.compile(r"seg_(\d+)")
    seg_indices = []
    for (_, _, detail) in features_df.columns:
        match = seg_pattern.search(detail)
        seg_indices.append(int(match.group(1)) if match else -1)

    n_segments = max(seg_indices) + 1
    segment_columns = [[] for _ in range(n_segments)]
    for idx, seg in enumerate(seg_indices):
        if seg >= 0:
            segment_columns[seg].append(idx)

    # Sort each segment's feature indices to ensure consistent feature order
    for seg in range(n_segments):
        segment_columns[seg].sort()

    # Now build the reshaped array
    n_features_per_segment = len(segment_columns[0])
    reshaped_array = np.zeros((n_epochs, n_features_per_segment, n_segments))

    for seg in range(n_segments):
        cols = segment_columns[seg]
        reshaped_array[:, :, seg] = features_df.iloc[:, cols].values

    return reshaped_array.transpose(0, 2, 1), labels

import torch

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # important for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_validate(model, device, train_loader, val_loader, optimizer, epoch, criterion):
    model.to(device)
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Compute training accuracy
        preds = output.argmax(dim=1)
        correct_train += (preds == target).sum().item()
        total_train += target.size(0)

    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            # Compute validation accuracy
            preds = output.argmax(dim=1)
            correct_val += (preds == target).sum().item()
            total_val += target.size(0)

    train_acc = correct_train / total_train
    val_acc = correct_val / total_val

    return (
        train_loss / len(train_loader),
        val_loss / len(val_loader),
        train_acc,
        val_acc
    )


def run_training(model_class, model_args, train_loader, val_loader, num_epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(*model_args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, val_loss, train_acc, val_acc = train_validate(
            model, device, train_loader, val_loader, optimizer, epoch, criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"[{model_class.__name__}] Epoch {epoch+1}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Train Acc = {train_acc:.2f}, Val Acc = {val_acc:.2f}")

    # Return losses, accuracies, and best model weights
    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_state