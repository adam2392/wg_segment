import os

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def generate_roc_curve(
    y_test, classifier_names, prediction_probs, fig_dir, reference, verbose=True
):
    """
    Plot ROC curves and report accompanying AUC.
    """

    fprs = []
    tprs = []

    fig, ax = plt.subplots(1, 1, dpi=300)

    for i, (name, proba) in enumerate(zip(classifier_names, prediction_probs)):
        if verbose:
            auc = roc_auc_score(y_test, proba[:, 1])
            print(f"{name} ROC auc score: {auc}")

        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])

        fprs.append(fpr)
        tprs.append(tpr)

        sns.lineplot(
            x=fpr, y=tpr, label=f"{name} ROC curve (area = {auc:.3f})", ci=None, ax=ax
        )
    if verbose:
        print()

    ax.set(
        title=f"ROC Curves for Random Forest Variants with {reference.capitalize()} Data",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    fig.tight_layout()

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fpath = os.path.join(fig_dir, f"roc_curves-{reference.lower()}.png")
    plt.savefig(fpath)

    return fprs, tprs


def generate_precision_recall_curve(
    y_test, classifier_names, prediction_probs, fig_dir, reference, verbose=True
):
    """
    Plot precision recall curves and report accompanying AUC.
    """

    precisions = []
    recalls = []

    fig, ax = plt.subplots(1, 1, dpi=300)

    for i, (name, proba) in enumerate(zip(classifier_names, prediction_probs)):
        precision, recall, _ = precision_recall_curve(y_test, proba[:, 1])

        precisions.append(precision)
        recalls.append(recall)

        if verbose:
            pr_auc = auc(recall, precision)
            print(f"{name} PR auc score: {pr_auc}")

        sns.lineplot(
            x=recall,
            y=precision,
            label=f"{name} precision-recall curve (area = {pr_auc:.3f})",
            ci=None,
            ax=ax,
        )

    if verbose:
        print()

    ax.set(
        title=f"Precision Recall Curves for Random Forest Variants with {reference.capitalize()} Data",
        xlabel="Recall",
        ylabel="Precision",
    )

    fig.tight_layout()

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fpath = os.path.join(fig_dir, f"pr_curves-{reference.lower()}.png")
    plt.savefig(fpath)

    return precisions, recalls
