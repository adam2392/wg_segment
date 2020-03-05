import collections
import re

import mne
import numpy as np

from natsort import natsorted
from random import randrange
from scipy.spatial import distance_matrix
from typing import Dict, List, Union


def _get_monopolar_data(raw, y_labels):
    """
    Construct signal time series with original monopolar channel signals.
    Corresponding white matter/grey matter labels are also returned.
    """
    ch_names = raw.ch_names

    X = raw.get_data()
    y = np.array([y_labels[ch] for ch in ch_names])

    return X, y, ch_names


def _find_bipolar_reference(ch_names):
    """
    Get corresponding lists of paired anode and cathode channels as well as
    any channels that are missing their neighboring channel.
    """
    ch_names = natsorted(ch_names)

    # get all unique electrodes
    elec_names = []
    for x in ch_names:

        elec_name = re.sub("[0-9]", "", x)

        if elec_name not in elec_names:
            elec_names.append(elec_name)

    # get the channel numbers for each electrode
    elec_to_channels = collections.defaultdict(list)
    for x in ch_names:
        elec, num = re.match("^([A-Za-z]+[']?)([0-9]+)$", x).groups()
        elec_to_channels[elec].append(num)

    # get bipolar reference
    anode_chs = []
    cathode_chs = []
    monopolar_chs = []

    for _elec_name, ch_list in elec_to_channels.items():

        n = len(ch_list)
        ch_list = np.array(ch_list)

        for (ch_num0, ch_num1) in zip(ch_list[0 : n - 1], ch_list[1:n]):

            if int(ch_num0) == int(ch_num1) - 1:

                anode_chs.append(f"{_elec_name}{ch_num0}")
                cathode_chs.append(f"{_elec_name}{ch_num1}")

            else:
                # Adjacent channel does not exist
                monopolar_chs.append(f"{_elec_name}{ch_num1}")

    return anode_chs, cathode_chs, monopolar_chs


def _get_bipolar_data(
    raw, y_labels, anode, cathode, monopolar=[], include_monopolar=False
):
    """
    Construct signal time series by subtracting adjacent channel signals.
    Corresponding white matter/grey matter labels for the anode channels are
    also returned.
    """
    anode_data = raw.get_data(anode)
    cathode_data = raw.get_data(cathode)
    mono_data = raw.get_data(monopolar) if len(monopolar) else None

    X = anode_data - cathode_data
    y = np.array([y_labels[ch] for ch in anode])
    ch_names = anode

    if include_monopolar and mono_data is not None:
        X = np.append(X, mono_data, axis=0)

        monopolar_labels = np.array([y_labels[ch] for ch in monopolar])
        y = np.append(y, monopolar_labels)
        ch_names += monopolar

    return X, y, ch_names


def _get_centered_data(raw: mne.io.Raw, y_labels: Dict, by_electrode=False):
    """
    Construct signal time series by subtracting mean signal for each electrode.
    Corresponding white matter/grey matter labels are also returned.
    """

    X = None
    y = None

    if not by_electrode:
        ch_names = raw.ch_names
        raw = raw.set_eeg_reference(ref_channels="average")

        X = raw.get_data()
        y = np.array([y_labels[ch] for ch in ch_names])

    else:
        sorted_y_labels = natsorted(y_labels.keys())

        # get all unique electrodes
        elec_names = []
        for x in sorted_y_labels:

            elec_name = re.sub("[0-9]", "", x)

            if elec_name not in elec_names:
                elec_names.append(elec_name)

        # get the channel numbers for each electrode
        elec_to_channels = collections.defaultdict(list)
        for x in sorted_y_labels:
            elec, num = re.match("^([A-Za-z]+[']?)([0-9]+)$", x).groups()
            elec_to_channels[elec].append(elec + str(num))

        # Get channel data by electrode and mean subtract them
        ch_names = []
        for elec in elec_to_channels:
            chans_list = elec_to_channels[elec]

            # Add channel names to ch_names
            ch_names += list(chans_list.keys())

            labels = [y_labels[ch] for ch in chans_list]
            labels = np.array(labels)

            # Subtract out average signal
            elec_data = raw.get_data(chans_list)
            avg_signal = np.mean(elec_data, axis=0)
            centered_data = elec_data - avg_signal

            if (X is None) or (y is None):
                X = centered_data
                y = labels
            else:
                X = np.append(X, centered_data, axis=0)
                y = np.append(y, labels)

    return X, y, ch_names


def _get_channel_labels(elec_descrip, ch_names):
    # Get binary class labels for each channel
    ch_labels = {}
    for i in range(len(elec_descrip["status_description"])):
        name = elec_descrip["name"][i]
        status = elec_descrip["status"][i]
        descrip = elec_descrip["status_description"][i]

        if name not in ch_names:
            continue

        if status == "bad" and descrip == "white matter":
            ch_labels[name] = 1
        elif name in ch_names:
            ch_labels[name] = 0

    return ch_labels


def _get_neighbors_of_channel(elec_to_chans: Dict, ch_pos, n_neighbors: int):
    """
    Return dictionary of channels mapped to its neighbors, including itself.
    
    Parameters
    ----------
        elec_to_chans: Dict[str: Dict[str: tuple(np.ndarray, np.ndarray)]]
            Dictionary of channels and tuple of signal data and label grouped
            by electrode.

        ch_pos: Dict[str: np.ndarray]]

        n_neighbors: int
            Number of neighbors to get for each channel.

    Returns
    -------
        chan_to_neighbors: Dict[str: List[str]]
            Mapping of channels to list of neighboring channel names.
    """
    chan_to_neighbors = collections.defaultdict(list)

    ch_names = list(ch_pos.keys())
    coords = list(ch_pos.values())
    dist_mat = distance_matrix(coords, coords)

    for i, name in enumerate(ch_pos):
        dists = dist_mat[i]

        # Dists will include the channel itself, so we get n_negihbors+1 channels
        closest = np.argsort(dists)[: n_neighbors + 1]
        chan_to_neighbors[name] = [ch_names[closest[i]] for i in range(len(closest))]

    return chan_to_neighbors


def _group_neighbors(
    X: np.ndarray, y: np.ndarray, ch_names: List[str], ch_pos: Dict, n_neighbors: int
):
    # Sort channels with data
    chan_to_data = dict(zip(ch_names, zip(X, y)))
    sorted_chs = natsorted(chan_to_data.items(), key=lambda x: x[0])
    sorted_chs = dict(sorted_chs)

    # Group channels by electrode to get neighbors
    elec_to_chans = collections.defaultdict(dict)
    for name, data in sorted_chs.items():
        elec, num = re.match("^([A-Za-z]+[']?)([0-9]+)$", name).groups()
        num = int(num)
        elec_to_chans[elec][num] = data

    chan_to_neighbors = _get_neighbors_of_channel(elec_to_chans, ch_pos, n_neighbors)

    chan_to_neighbor_data = collections.defaultdict(list)
    for chan, neighbors in chan_to_neighbors.items():
        neighbor_data = np.array(
            [X for ch, (X, _) in sorted_chs.items() if ch in neighbors]
        )

        # Missing a neighbor!
        if not neighbor_data.shape[0] == n_neighbors + 1:
            continue

        ylabel = sorted_chs[chan][1]
        chan_to_neighbor_data[chan] = (neighbor_data, ylabel)

    X_neighbors = np.array([data for data, _ in chan_to_neighbor_data.values()])
    y_neighbors = np.array([label for _, label in chan_to_neighbor_data.values()])
    ch_names_neighbors = list(chan_to_neighbor_data.keys())

    return X_neighbors, y_neighbors, ch_names_neighbors


def get_data_from_raw(
    raw: mne.io.Raw,
    ch_pos: Dict,
    elec_descrip: Dict,
    window_size_seconds: int = 10,
    reference: str = "monopolar",
    n_neighbors: int = 0,
    strided: bool = True,
    *args,
    **kwargs,
):
    """
    Get time series data and corresponding white matter/grey matter labels
    from Raw object.
    """
    ch_names = raw.ch_names
    n_samples = int(window_size_seconds * raw.info["sfreq"])

    ch_descrips = _get_channel_labels(elec_descrip, ch_names)

    reference = reference.lower()
    if reference == "monopolar":
        X, y, ch_names = _get_monopolar_data(raw, ch_descrips)

    elif reference == "bipolar":
        include_monopolar = (
            kwargs["include_monopolar"] if "include_monopolar" in kwargs else True
        )

        anode, cathode, mono = _find_bipolar_reference(ch_names)
        X, y, ch_names = _get_bipolar_data(
            raw, ch_descrips, anode, cathode, mono, include_monopolar
        )

    elif reference == "mean-subtracted":
        by_electrode = kwargs["by_electrode"] if "by_electrode" in kwargs else False
        X, y, ch_names = _get_centered_data(raw, ch_descrips, by_electrode)

    else:
        raise ValueError(f"Reference type {reference} not recognized")

    # If strided, construct each sample with random windows of each channel,
    # otherwise just use the same window for all channels
    if not strided:
        start = randrange(raw.n_times - n_samples + 1)
        stop = start + n_samples
        X = X[:, start:stop]
    else:
        m, n = X.shape
        start = np.random.randint(low=0, high=raw.n_times - n_samples + 1, size=m)
        stop = start + n_samples

        # Get random time windows for each row
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = X.strides
        windows = strided(
            X, shape=(m, n - n_samples + 1, n_samples), strides=(s0, s1, s1)
        )

        X = windows[np.arange(len(start)), start]

    if n_neighbors > 0:
        X, y, ch_names = _group_neighbors(X, y, ch_names, ch_pos, n_neighbors)

    print(f"Data shapes: X = {X.shape}, y = {y.shape}")

    return X, y, ch_names
