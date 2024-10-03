import warnings
from collections.abc import Iterable
from itertools import product
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.decomposition import NMF
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import json

CLASS_NAMES = {
    0: "BACKGROUND",
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}

def get_class_count(Y0):
    class_count = np.bincount(Y0[:,::4,::4,1].ravel())
    try:
        import pandas as pd
        df = pd.DataFrame(class_count, index=CLASS_NAMES.values(), columns=["counts"])
        df = df.drop("BACKGROUND")
        df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
        print(df)
    except ModuleNotFoundError:
        print("install 'pandas' to show class counts")
    return class_count

def cls_dict_from_label(y, y_class):
    return dict(
        (r.label, int(np.median(y_class[r.slice][y[r.slice] == r.label]))) for r in regionprops(y)
    )


def get_data(path, n=None, shuffle=True, normalize=False, seed=None):
    rng = np.random if seed is None else np.random.RandomState(seed)

    path = Path(path)
    X = np.load(path / "images.npy")
    Y0 = np.load(path / "labels.npy")
    assert len(X) == len(Y0)

    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    idx = idx[:n]

    X = X[idx]
    Y0 = Y0[idx]

    if normalize:
        X = (X / 255).astype(np.float32)

    Y = Y0[..., 0]
    D = np.array(
        [
            cls_dict_from_label(y, y_class)
            for y, y_class in tqdm(zip(Y0[..., 0], Y0[..., 1]), total=len(Y0))
        ]
    )

    return X, Y, D, Y0, idx

X, Y, D, Y0, idx = get_data('/dataset/CoNIC', seed=322)

X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv = train_test_split(X, Y, D, Y0, idx, test_size=0.1, random_state=322)
class_count = get_class_count(Y0)


def oversample_classes(X, Y, D, Y0, idx, n_extra_classes=4, seed=None):
    rng = np.random if seed is None else np.random.RandomState(seed)

    # get the most infrequent classes
    class_counts = np.bincount(Y0[:, ::4, ::4, 1].ravel(), minlength=len(CLASS_NAMES))
    extra_classes = np.argsort(class_counts)[:n_extra_classes]
    all(
        class_counts[c] > 0 or print(f"count 0 for class {c} ({CLASS_NAMES[c]})")
        for c in extra_classes
    )

    # how many extra samples (more for infrequent classes)
    n_extras = np.sqrt(np.sum(class_counts[1:]) / class_counts[extra_classes])
    
    n_extras = n_extras / np.max(n_extras)
    print("oversample classes", extra_classes)
    idx_take = np.arange(len(X))

    for c, n_extra in zip(extra_classes, n_extras):
        # oversample probability is ~ number of instances
        prob = np.sum(Y0[:, ::2, ::2, 1] == c, axis=(1, 2))
        prob = np.clip(prob, 0, np.percentile(prob, 99.8))
        prob = prob ** 2
        # prob[prob<np.percentile(prob,90)] = 0
        prob = prob / np.sum(prob)
        n_extra = int(n_extra * len(X))
        print(f"adding {n_extra} images of class {c} ({CLASS_NAMES[c]})")
        idx_extra = rng.choice(np.arange(len(X)), n_extra, p=prob)
        idx_take = np.append(idx_take, idx_extra)

    X, Y, D, Y0, idx = map(lambda x: x[idx_take], (X, Y, D, Y0, idx))
    return X, Y, D, Y0, idx


X, Y, D, Y0, idx = oversample_classes(X, Y, D, Y0, idx, seed=322)