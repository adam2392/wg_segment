import collections
import os
import re
import sys

sys.path.append("../..")

import numpy as np
import mne

from mne import io
from mpl_toolkits.mplot3d import Axes3D
from natsort import natsorted
from pathlib import Path
from random import randrange
from rerf.rerfClassifier import rerfClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Dict, List, Union

from random_forest_visualization import generate_roc_curve, generate_precision_recall_curve
from mne_bids.utils import print_dir_tree, _parse_bids_filename
from mne_bids import write_raw_bids, read_raw_bids, make_bids_basename, make_bids_folders
from mne_bids.tsv_handler import _from_tsv, _to_tsv

from sample_scripts import file_utils


# sns.set_context("paper", font_scale=1.5)


def load_elecs_data(elecfile: Union[str, Path]):
    """
    Load each brain image scan as a NiBabel image object.

    Parameters
    ----------
        elecfile: Union[str, Path]
            Space-delimited text file of contact labels and contact
            coordinates in mm space.

    Returns
    -------
        elecinitfile: dict(label: coord)
            A dictionary of contact coordinates in mm space. Keys are
            individual contact labels, and values are the corresponding
            coordinates in mm space.
    """

    eleccoords_mm = {}

    elecfile = str(elecfile)

    if elecfile.endswith(".txt"):
        with open(elecfile) as f:
            for l in f:
                row = l.split()
                if len(row) == 4:
                    eleccoords_mm[row[0]] = np.array(list(map(float, row[1:])))
                elif len(row) == 6:
                    eleccoords_mm[row[1]] = np.array(list(map(float, row[2:5])))
                else:
                    raise ValueError(
                        "Unrecognized electrode coordinate text format"
                    )
    else:
        matreader = MatReader()
        data = matreader.loadmat(elecfile)

        eleclabels = data["eleclabels"]
        elecmatrix = data["elecmatrix"]
        print(f"Electrode matrix shape: {elecmatrix.shape}")

        for i in range(len(eleclabels)):
            eleccoords_mm[eleclabels[i][0].strip()] = elecmatrix[i]

    print(f'Electrode labels: {eleccoords_mm.keys()}')

    return eleccoords_mm


def make_seeg_montage_edf(subject: str, edf_fpath: str, elecs_fpath: str):
    """
    Construct a SEEG montage from .edf data.
    """
    missing_chs = []

    if subject == "la03":
        missing_chs = ["Y16", "R'1", "R'2", "R'3", "R'4",
                       "R'5", "R'6", "R'7", "R'8", "R'9",
                       "X'1", "X'2", "X'9", "X'10"]  # for la03

    # Read .edf file into mne Raw object
    raw = io.read_raw_edf(edf_fpath, preload=True, verbose=False)

    # Setting channel types
    raw = file_utils._set_channel_types(raw, verbose=False)

    # Standardizing channel labels
    raw = file_utils.channel_text_scrub(raw)

    # Scrub out bad channels
    chan_scrub = file_utils.ChannelScrub()
    bad_chs = chan_scrub.look_for_bad_channels(raw.ch_names)
    raw.info["bads"] = bad_chs + missing_chs

    # Exclude bad channels from raw object
    raw.pick_types(eeg=True, exclude="bads")

    # Construct montage
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mri")

    return raw, montage


def find_bipolar_reference(ch_names):
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

        for (ch_num0, ch_num1) in zip(ch_list[0:n-1], ch_list[1:n]):

            if int(ch_num0) == int(ch_num1) - 1:

                anode_chs.append(f"{_elec_name}{ch_num0}")
                cathode_chs.append(f"{_elec_name}{ch_num1}")

            else:
                # Adjacent channel does not exist
                monopolar_chs.append(f"{_elec_name}{ch_num1}")

    return anode_chs, cathode_chs, monopolar_chs


def get_monopolar_data(raw, ch_labels):
    """
    Construct signal time series with original monopolar channel signals.
    Corresponding white matter/grey matter labels are also returned.
    """
    ch_names = raw.ch_names

    X = raw.get_data()
    y = np.array([ch_labels[ch] for ch in ch_names])

    return X, y


def get_bipolar_data(
    raw,
    ch_labels,
    anode,
    cathode,
    monopolar=[],
    include_monopolar=True
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


def get_averaged_data(raw: mne.io.Raw, ch_labels: Dict, by_electrode=False):
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
    **kwargs
):
    """
    Get time series data and corresponding white matter/grey matter labels
    from Raw object.
    """
    ch_names = raw.ch_names
    reference = reference.lower()
    n_samples = int(window_size * raw.info["sfreq"])

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
        X, y = get_monopolar_data(raw, ch_labels)

    elif reference == "bipolar":
        include_monopolar = True
        if "include_monopolar" in kwargs:
            include_monopolar = kwargs["include_monopolar"]

        anode, cathode, mono = find_bipolar_reference(ch_names)
        X, y = get_bipolar_data(raw, ch_labels, anode, cathode, mono, include_monopolar)

    elif reference == "averaged":
        by_electrode = False
        if "by_electrode" in kwargs:
            by_electrode = kwargs["by_electrode"]

        X, y = get_averaged_data(raw, ch_labels, by_electrode)

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
        start = np.random.randint(
            low=0,
            high=raw.n_times-n_samples+1,
            size=m
        )
        stop = start + n_samples

        # Get random time windows for each row
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = X.strides
        windows = strided(
            X, shape=(m, n-n_samples+1, n_samples), strides=(s0, s1, s1)
        )

        X = windows[np.arange(len(start)), start]

    print(f"Data dimensions - X.shape {X.shape}, y.shape {y.shape}")

    return X, y


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_names: List[str],
    *args,
    **kwargs
):
    """
    Train each classifier.
    """

    clfs = []
    n_samples = int(kwargs["window_size"] * kwargs["sfreq"])

    for clfname in classifier_names:
        name = clfname.lower()
        clf = None

        if name in ["rf", "random forest", "standard random forest"]:
            clf = RandomForestClassifier(n_estimators=500)

        elif name in ["rerf", "random project random forest"]:
            clf = rerfClassifier(n_estimators=500)

        elif name in ["srerf", "sporf", "structured rerf"]:
            clf = rerfClassifier(
                projection_matrix="S-RerF",
                n_estimators=500,
                image_height=1,
                image_width=n_samples,
                patch_height_max=1,
                patch_height_min=1
            )

        if clf is not None:
            clfs.append(clf)

    for clf in clfs:
        clf.fit(X_train, y_train)

    fitted_clfs = dict(zip(classifier_names, clfs))

    return fitted_clfs


def predict_classifiers(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifiers: Dict,
    verbose=True
) -> List[np.ndarray]:
    """
    Predict and evaluate accuracy for each classifier.
    """

    predictions = []
    prediction_probs = []

    zero_pred = np.zeros(y_test.shape)
    zero_pred_proba = np.array([[1, 0] for _ in range(len(y_test))])
    predictions.append(zero_pred)
    prediction_probs.append(zero_pred_proba)

    if verbose:
        print(f"Zero classifier accuracy: {accuracy_score(y_test, zero_pred)}")

    for name, clf in classifiers.items():

        ypred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        predictions.append(ypred)
        prediction_probs.append(proba)

        if verbose:
            print(f"{name} accuracy: {accuracy_score(y_test, ypred)}")
    print()

    return predictions, prediction_probs


def evaluate_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_names: List[str],
    window_size=10,
    sfreq=2000
):
    """
    Wrapper method that calls train and predict. Evaluates
    classifiers with accuracy.
    """
    fitted_clfs = train_classifiers(
        X_train,
        y_train,
        classifier_names,
        window_size=window_size,
        sfreq=sfreq
    )

    predictions, prediction_probs = predict_classifiers(
        X_test,
        y_test,
        fitted_clfs,
        verbose=True
    )

    return predictions, prediction_probs


if __name__ == "__main__":
    np.random.seed(2)

    bids_root = "/workspaces/ChesterHuynh/research/data/bids_layout_data/"
    # bids_root = "/Users/ChesterHuynh/research/data/bids_layout_data/"
    patid = "la03"
    ses = "interictal"
    task = "monitor"
    acq = "seeg"
    run = "01"

    sourcedata = os.path.join(bids_root, "sourcedata", patid)
    derivatives = os.path.join(bids_root, "derivatives", "freesurfer", patid)

    pat_dir = os.path.join(bids_root, f"sub-{patid}")
    derivatives_meta_dir = os.path.join(derivatives, f"ses-{ses}", "ieeg")
    elec_meta_dir = os.path.join(
        bids_root, f"sub-{patid}", f"ses-{ses}", "ieeg"
    )

    edf_dir = os.path.join(sourcedata, acq, "edf")
    fif_dir = os.path.join(sourcedata, acq, "fif")
    elecs_dir = os.path.join(derivatives, "elecs")

    # Get relevant file paths
    elec_layout_fpath = file_utils._get_pt_electrode_layout(sourcedata, patid)

    edf_fpath = os.path.join(edf_dir, f"{patid}_inter.edf")
    elec_fpath = os.path.join(elecs_dir, f"{patid}_elecxyz.txt")

    elec_descrip_fname = make_bids_basename(
        subject=patid,
        session=ses,
        task=task,
        acquisition=acq,
        run=run,
        suffix="channels.tsv"
    )

    elec_descrip_fpath = os.path.join(elec_meta_dir, elec_descrip_fname)

    ch_pos = load_elecs_data(elec_fpath)
    elec_descrip = _from_tsv(elec_descrip_fpath)

    raw, montage = make_seeg_montage_edf(patid, edf_fpath, elec_fpath)
    raw = raw.set_montage(montage)

    # Data processing
    window_sizes = [5, 10, 15]  # in seconds
    references = ["averaged", "bipolar", "monopolar"]

    for window_size in window_sizes:
        for reference in references:
            X, y = get_data_from_raw(
                raw,
                elec_descrip,
                window_size_seconds=window_size,
                strided=True,
                reference=reference,
                by_electrode=False,
                include_monopolar=False
            )

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
            classifier_names = ["Random Forest", "RerF", "SPORF"]
            predictions, prediction_probs = evaluate_classifiers(
                X_train, y_train,
                X_test, y_test,
                classifier_names,
                window_size=window_size,
                sfreq=raw.info["sfreq"]
            )

            fig_dir = f"/workspaces/ChesterHuynh/research/seeg localization/figs/{window_size}sec/"

            classifier_names = ["Zero"] + classifier_names

            fprs, tprs = generate_roc_curve(
                y_test,
                classifier_names,
                prediction_probs,
                fig_dir,
                reference
            )

            precisions, recalls = generate_precision_recall_curve(
                y_test,
                classifier_names,
                prediction_probs,
                fig_dir,
                reference
            )
