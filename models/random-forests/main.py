import collections
import os
import re
import sys

import numpy as np
import mne

from joblib import dump, load
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

# sys.path.append("/Users/ChesterHuynh/research/seeg localization/")
sys.path.append("/workspaces/ChesterHuynh/research/seeg localization/")

from data_wrangler import get_data_from_raw
from mne_bids.utils import print_dir_tree, _parse_bids_filename
from mne_bids import (
    write_raw_bids,
    read_raw_bids,
    make_bids_basename,
    make_bids_folders,
)
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from sample_scripts import file_utils
from visualization import (
    generate_roc_curve,
    generate_precision_recall_curve,
)


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
                    raise ValueError("Unrecognized electrode coordinate text format")
    else:
        matreader = MatReader()
        data = matreader.loadmat(elecfile)

        eleclabels = data["eleclabels"]
        elecmatrix = data["elecmatrix"]
        print(f"Electrode matrix shape: {elecmatrix.shape}")

        for i in range(len(eleclabels)):
            eleccoords_mm[eleclabels[i][0].strip()] = elecmatrix[i]

    print(f"Electrode labels: {eleccoords_mm.keys()}")

    return eleccoords_mm


def make_seeg_montage_edf(subject: str, edf_fpath: str, elecs_fpath: str):
    """
    Construct a SEEG montage from .edf data.
    """
    missing_chs = []

    if subject == "la03":
        missing_chs = [
            "Y16",
            "R'1",
            "R'2",
            "R'3",
            "R'4",
            "R'5",
            "R'6",
            "R'7",
            "R'8",
            "R'9",
            "X'1",
            "X'2",
            "X'9",
            "X'10",
        ]  # for la03

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


def _drop_non_wm_bad_channels(raw, elec_descriptions):
    ch_names = raw.ch_names

    bad_chs = []
    for i in range(len(elec_descriptions["status_description"])):
        name = elec_descriptions["name"][i]
        status = elec_descriptions["status"][i]
        descrip = elec_descriptions["status_description"][i]

        if (status == "bad") and (descrip != "white matter"):
            bad_chs.append(name)

    raw.drop_channels(bad_chs)

    return raw


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_names: List[str],
    *args,
    **kwargs,
):
    """
    Train each classifier.
    """

    clfs = []
    n_samples = int(kwargs["window_size"] * kwargs["sfreq"])

    n_channels, *dims = X_train.shape
    X_vectorized = X_train.reshape(n_channels, np.prod(dims))
    if len(dims) > 1:
        n_elecs = dims[0]
    else:
        n_elecs = 1

    for clfname in classifier_names:
        name = clfname.lower()
        clf = None

        if name in ["rf", "random forest", "standard random forest"]:
            clf = RandomForestClassifier(random_state=1)

        elif name in ["sporf"]:
            clf = rerfClassifier(random_state=1)

        elif name in ["morf", "structured rerf"]:
            clf = rerfClassifier(
                projection_matrix="S-RerF",
                random_state=1,
                image_height=n_elecs,
                image_width=n_samples,
                patch_height_max=1,
                patch_height_min=1,
            )
        else:
            raise ValueError(f"Classifier name {name} not recognized")

        if clf is not None:
            clfs.append(clf)

    for name, clf in zip(classifier_names, clfs):
        clf.fit(X_vectorized, y_train)

    fitted_clfs = dict(zip(classifier_names, clfs))

    return fitted_clfs


def predict_classifiers(
    X_test: np.ndarray, y_test: np.ndarray, classifiers: Dict, verbose=True
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
        n_channels, *dims = X_test.shape
        X_vectorized = X_test.reshape(n_channels, np.prod(dims))

        ypred = clf.predict(X_vectorized)
        proba = clf.predict_proba(X_vectorized)

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
    sfreq=2000,
    *args,
    **kwargs,
):
    """
    Wrapper method that calls train and predict. Evaluates
    classifiers with accuracy.
    """
    fitted_clfs = train_classifiers(
        X_train, y_train, classifier_names, window_size=window_size, sfreq=sfreq
    )

    predictions, prediction_probs = predict_classifiers(
        X_test, y_test, fitted_clfs, verbose=True
    )

    persist = kwargs["persist"] if "persist" in kwargs else True

    if persist:
        persist_dir = kwargs["persist_dir"] if "persist_dir" in kwargs else None
        if not persist_dir:
            raise KeyError("No file path specified for persisting classifiers")

        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        for name, clf in fitted_clfs.items():
            persist_path = os.path.join(persist_dir, f"{name}.joblib")
            dump(clf, persist_path)

    return predictions, prediction_probs


if __name__ == "__main__":
    np.random.seed(1)

    patid = "la03"
    ses = "interictal"
    task = "monitor"
    acq = "seeg"
    run = "01"

    # bids_root = "/Users/ChesterHuynh/research/data/bids_layout_data/"
    bids_root = "/workspaces/ChesterHuynh/research/data/bids_layout_data/"

    sourcedata = os.path.join(bids_root, "sourcedata", patid)
    derivatives = os.path.join(bids_root, "derivatives", "freesurfer", patid)

    pat_dir = os.path.join(bids_root, f"sub-{patid}")
    derivatives_meta_dir = os.path.join(derivatives, f"ses-{ses}", "ieeg")

    elec_meta_dir = os.path.join(bids_root, f"sub-{patid}", f"ses-{ses}", "ieeg")
    elecs_dir = os.path.join(derivatives, "elecs")

    # Get relevant file paths
    elec_layout_fpath = file_utils._get_pt_electrode_layout(sourcedata, patid)

    # Get channel positions
    elec_fpath = os.path.join(elecs_dir, f"{patid}_elecxyz.txt")
    ch_pos = load_elecs_data(elec_fpath)

    # # Get .edf file and use to set montage
    # edf_dir = os.path.join(sourcedata, acq, "edf")
    # edf_fpath = os.path.join(edf_dir, f"{patid}_inter.edf")
    # raw, montage = make_seeg_montage_edf(patid, edf_fpath, elec_fpath)
    # raw = raw.set_montage(montage)

    elec_descrip_fname = make_bids_basename(
        subject=patid,
        session=ses,
        task=task,
        acquisition=acq,
        run=run,
        suffix="channels.tsv",
    )

    eeg_data = make_bids_basename(
        subject=patid,
        session=ses,
        task=task,
        acquisition=acq,
        run=run,
        suffix="ieeg.vhdr",
    )

    # Read channel descriptions
    elec_descrip_fpath = os.path.join(elec_meta_dir, elec_descrip_fname)
    elec_descriptions = _from_tsv(elec_descrip_fpath)

    # Get raw object
    extra_params = dict(preload=True)
    raw = read_raw_bids(eeg_data, bids_root, extra_params=extra_params)

    # Drop non-wm channels and non-seeg channels
    raw = _drop_non_wm_bad_channels(raw, elec_descriptions)
    raw.pick_types(seeg=True, exclude=[])

    # Data processing
    window_sizes = [5, 10, 15]  # in seconds
    references = [
        "mean-subtracted",
        # "bipolar",
        "monopolar",
    ]

    for window_size in window_sizes:
        for reference in references:
            X, y, _ = get_data_from_raw(
                raw,
                ch_pos,
                elec_descriptions,
                window_size_seconds=window_size,
                n_neighbors=0,
                reference=reference,
                strided=False,
                by_electrode=False,
                include_monopolar=False,
            )

            # Split dataset into training set and test set
            print(f"Time window: {window_size} sec, reference: {reference}")
            print("------------------------------------------------------")

            results_dir = f"/workspaces/ChesterHuynh/research/seeg localization/results/{patid}/{window_size}sec/"

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            classifier_names = ["Random Forest", "SPORF", "MORF"]
            predictions, prediction_probs = evaluate_classifiers(
                X_train,
                y_train,
                X_test,
                y_test,
                classifier_names,
                window_size=window_size,
                sfreq=raw.info["sfreq"],
                persist=False,
                # persist_dir=results_dir
            )

            fig_dir = f"/workspaces/ChesterHuynh/research/seeg localization/figs/{patid}/{window_size}sec/"

            classifier_names = ["Zero"] + classifier_names

            fprs, tprs = generate_roc_curve(
                y_test, classifier_names, prediction_probs, fig_dir, reference
            )

            precisions, recalls = generate_precision_recall_curve(
                y_test, classifier_names, prediction_probs, fig_dir, reference
            )
