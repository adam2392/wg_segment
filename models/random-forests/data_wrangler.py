import collections
import re

import numpy as np

from natsort import natsorted
from random import randrange

def _get_monopolar_data(raw, ch_labels):
    """
    Construct signal time series with original monopolar channel signals.
    Corresponding white matter/grey matter labels are also returned.
    """
    ch_names = raw.ch_names

    X = raw.get_data()
    y = np.array([ch_labels[ch] for ch in ch_names])

    return X, y


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
    raw, ch_labels, anode, cathode, monopolar=[], include_monopolar=True
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
    y = np.array([ch_labels[ch] for ch in anode])

    if mono_data is not None and include_monopolar:
        X = np.append(X, mono_data, axis=0)

        mono_labels = np.array([ch_labels[ch] for ch in monopolar])
        y = np.append(y, mono_labels)

    return X, y


def _get_averaged_data(raw: mne.io.Raw, ch_labels: Dict, by_electrode=False):
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
        y = np.array([ch_labels[ch] for ch in ch_names])
    else:
        ch_names = natsorted(ch_labels.keys())

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
            elec_to_channels[elec].append(elec + str(num))

        for elec in elec_to_channels:
            chans_list = elec_to_channels[elec]

            labels = [ch_labels[ch] for ch in chans_list]
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

    return X, y


def get_data_from_raw(
    raw: mne.io.Raw,
    elec_descrip: Dict,
    window_size_seconds: int = 10,
    strided: bool = True,
    reference: str = "monopolar",
    *args,
    **kwargs,
):
    """
    Get time series data and corresponding white matter/grey matter labels
    from Raw object.
    """
    ch_names = raw.ch_names
    reference = reference.lower()
    n_samples = int(window_size_seconds * raw.info["sfreq"])

    # Get binary class labels for each channel
    ch_labels = {}
    for i in range(len(elec_descrip["status_description"])):
        name = elec_descrip["name"][i]
        status = elec_descrip["status"][i]
        descrip = elec_descrip["status_description"][i]

        if status == "bad" and descrip == "white matter":
            ch_labels[name] = 1
        elif name in ch_names:
            ch_labels[name] = 0

    if reference == "monopolar":
        X, y = _get_monopolar_data(raw, ch_labels)

    elif reference == "bipolar":
        include_monopolar = True
        if "include_monopolar" in kwargs:
            include_monopolar = kwargs["include_monopolar"]

        anode, cathode, mono = _find_bipolar_reference(ch_names)
        X, y = _get_bipolar_data(raw, ch_labels, anode, cathode, mono, include_monopolar)

    elif reference == "averaged":
        by_electrode = False
        if "by_electrode" in kwargs:
            by_electrode = kwargs["by_electrode"]

        X, y = _get_averaged_data(raw, ch_labels, by_electrode)

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

    print(f"Data dimensions - X.shape {X.shape}, y.shape {y.shape}")

    return X, y
