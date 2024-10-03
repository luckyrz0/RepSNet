import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from augmend.transforms import *
# from augmend.transforms import *
# from csbdeep.utils import _raise
from skimage.draw import polygon
from skimage.measure import regionprops
from sklearn.decomposition import NMF
from tqdm.auto import tqdm

CLASS_NAMES = {
    0: "BACKGROUND",
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}

def _get_global_rng():
    return np.random.random.__self__

def _validate_rng(rng):
    if rng is None or rng is np.random:
        rng = _get_global_rng()
    return rng

def cls_dict_from_label(y, y_class):
    return dict(
        (r.label, int(np.median(y_class[r.slice][y[r.slice] == r.label]))) for r in regionprops(y)
    )


def get_data(path, n=None, shuffle=True, normalize=True, seed=None):
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


### AUGMENTATIONS


def _assert_uint8_image(x):
    assert x.ndim == 3 and x.shape[-1] == 3 and x.dtype.type is np.uint8


def rgb_to_density(x):
    _assert_uint8_image(x)
    x = np.maximum(x, 1)
    return np.maximum(-1 * np.log(x / 255), 1e-6)


def density_to_rgb(x):
    return np.clip(255 * np.exp(-x), 0, 255).astype(np.uint8)


def rgb_to_lab(x):
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)


def lab_to_rgb(x):
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_LAB2RGB)


def extract_stains(x, subsample=128, l1_reg=0.001, tissue_threshold=200):
    """Non-negative matrix factorization 
    
    Let x be the image as optical densities with shape (N,3) 

    then we want to decompose it as 

    x = W * H 

    with
        W: stain values of shape (N, 2)
        H: staining matrix of shape (2, 3) 
        
    Solve it as 
    
    min (x - W * H)^2 + |H|_1 

    with additonal sparsity prior on the stains W 
    """
    _assert_uint8_image(x)

    model = NMF(
        n_components=2, init="random", random_state=0, alpha_W=l1_reg, alpha_H=0, l1_ratio=1
    )

    # optical density
    density = rgb_to_density(x)

    # only select darker regions
    tissue_mask = rgb_to_lab(x)[..., 0] < tissue_threshold

    values = density[tissue_mask]

    # compute stain matrix on subsampled values (way faster)
    model.fit(values[::subsample])

    H = model.components_

    # normalize rows
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    if H[0, 0] < H[1, 0]:
        H = H[[1, 0]]

    # get stains on full image
    Hinv = np.linalg.pinv(H)
    stains = density.reshape((-1, 3)) @ Hinv
    stains = stains.reshape(x.shape[:2] + (2,))

    return H, stains


def stains_to_rgb(stains, stain_matrix):
    assert stains.ndim == 3 and stains.shape[-1] == 2
    assert stain_matrix.shape == (2, 3)
    return density_to_rgb(stains @ stain_matrix)


def augment_stains(x, amount_matrix=0.2, amount_stains=0.2, n_samples=1, subsample=128, rng=None):
    """ 
    create stain color augmented versions of x by 
    randomly perturbing the stain matrix by given amount

    1) extract stain matrix M and associated stains
    2) add uniform random noise (+- scale) to stain matrix
    3) reconstruct image 
    """
    _assert_uint8_image(x)
    if rng is None:
        rng = np.random

    M, stains = extract_stains(x, subsample=subsample)

    M = np.expand_dims(M, 0) + amount_matrix * rng.uniform(-1, 1, (n_samples, 2, 3))
    M = np.maximum(M, 0)

    stains = np.expand_dims(stains, 0) * (
        1 + amount_stains * rng.uniform(-1, 1, (n_samples, 1, 1, 2))
    )
    stains = np.maximum(stains, 0)

    if n_samples == 1:
        return stains_to_rgb(stains[0], M[0])
    else:
        return np.stack(tuple(stains_to_rgb(s, m) for s, m in zip(stains, M)), 0)


class HEStaining(BaseTransform):
    """HE staining augmentations"""

    @staticmethod
    def _augment(x, rng, amount_matrix, amount_stains):
        rng = _validate_rng(rng)
        # x_rgb = (255 * np.clip(x, 0, 1)).astype(np.uint8)
        x_rgb = x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = augment_stains(
                    x_rgb,
                    amount_matrix=amount_matrix,
                    amount_stains=amount_stains,
                    subsample=128,
                    n_samples=1,
                    rng=rng,
                )
            except:
                res = x_rgb
        return res 

    def __init__(self, amount_matrix=0.15, amount_stains=0.4):
        super().__init__(
            default_kwargs=dict(amount_matrix=amount_matrix, amount_stains=amount_stains),
            transform_func=self._augment,
        )


class HueBrightnessSaturation(BaseTransform):
    """apply affine intensity shift where background is bright"""

    @staticmethod
    def hbs_adjust(x, rng, hue, brightness, saturation):
        # print(x.shape)
        # x = transforms.ToPILImage()(torch.from_numpy(x))
        x = torch.tensor(x)
        x = x.permute(2,1,0)
        def _prep(s, negate=True):
            s = (-s if negate else s, s) if np.isscalar(s) else tuple(s)
            assert len(s) == 2
            return s
        
        hue = _prep(hue)
        brightness = _prep(brightness)
        saturation = _prep(saturation, False)
        # assert x.ndim == 3 and x.shape[-1] == 3
        rng = _validate_rng(rng)
        h_hue = rng.uniform(*hue)
        h_brightness = rng.uniform(*brightness)
        h_saturation = rng.uniform(*saturation)
        
        x = transforms.functional.adjust_hue(x, h_hue)
        x = transforms.functional.adjust_brightness(x, abs(h_brightness))
        x = transforms.functional.adjust_saturation(x, h_saturation)

        x = x.permute(2,1,0)
        # print(x.shape)
        return  x.numpy()

    def __init__(self, hue=0.1, brightness=0, saturation=1):
        """
        hue:        add +- value
        brightness: add +- value
        saturation: multiply by value 
        -> set hue=0, brightness=0, saturation=1 for no effect
        """
        super().__init__(
            default_kwargs=dict(hue=hue, brightness=brightness, saturation=saturation),
            transform_func=self.hbs_adjust,
        )


### PREDICTION


def refine(labels, polys, thr=0.5, w_winner=2, progress=False):
    """shape refinement"""
    thr = float(thr)
    assert 0 <= thr <= 1, f"required: 0 <= {thr} <= 1"
    if thr == 1:
        # to include only pixels where all polys agree
        # because we take mask > thr below
        thr -= np.finfo(float).eps
    nms = polys["nms"]
    obj_ind = np.flatnonzero(nms["suppressed"] == -1)
    assert np.allclose(nms["scores"][obj_ind], sorted(nms["scores"][obj_ind])[::-1])
    mask = np.zeros_like(labels)
    # mask_soft = np.zeros_like(labels, float)

    # TODO: use prob/scores for weighting?
    # TODO: use mask that weights pixels on distance to poly boundary?
    for k, i in tqdm(
        zip(range(len(obj_ind), 0, -1), reversed(obj_ind)),
        total=len(obj_ind),
        disable=(not progress),
    ):
        polys_i = nms["coord"][i : i + 1]  # winner poly after nms
        polys_i_suppressed = nms["coord"][nms["suppressed"] == i]  # suppressed polys by winner
        # array of all polys (first winner, then all suppressed)
        polys_i = np.concatenate([polys_i, polys_i_suppressed], axis=0)
        # bounding slice around all polys wrt image
        ss = tuple(
            slice(max(int(np.floor(start)), 0), min(int(np.ceil(stop)), w))
            for start, stop, w in zip(
                np.min(polys_i, axis=(0, 2)), np.max(polys_i, axis=(0, 2)), labels.shape
            )
        )
        # shape of image crop/region that contains all polys
        shape_i = tuple(s.stop - s.start for s in ss)
        # offset of image region
        offset = np.array([s.start for s in ss]).reshape(2, 1)
        # voting weights for polys
        n_i = len(polys_i)
        # vote weight of winning poly (1 = same vote as each suppressed poly)
        weight_winner = w_winner
        # define and normalize weights for all polys
        polys_i_weights = np.ones(n_i)
        polys_i_weights[0] = weight_winner
        # polys_i_weights = np.array([weight_winner if j==0 else max(0,n_i-weight_winner)/(n_i-1) for j in range(n_i)])
        polys_i_weights = polys_i_weights / np.sum(polys_i_weights)
        # display(polys_i_weights)
        assert np.allclose(np.sum(polys_i_weights), 1)
        # merge by summing weighted poly masks
        mask_i = np.zeros(shape_i, float)
        for p, w in zip(polys_i, polys_i_weights):
            ind = polygon(*(p - offset), shape=shape_i)
            mask_i[ind] += w
        # write refined shape for instance i back to new label image
        # refined shape are all pixels with accumulated votes >= threshold
        mask[ss][mask_i > thr] = k
        # mask_soft[ss][mask_i>0] += mask_i[mask_i>0]

    return mask  # , mask_soft


def flip(x, doit=True, reverse=True):
    """Flip stardist cnn predictions."""
    assert x.ndim in (2, 3)
    if not doit:
        return x
    if x.ndim == 2 or reverse == False:
        return np.flipud(x)
    # dist image has radial distances as 3rd dimension
    # -> need to reverse values
    z = np.flipud(x)
    z = np.concatenate((z[..., 0:1], z[..., :0:-1]), axis=-1)
    return z


def crop_center(x, crop_shape):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
    w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
    x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def count_classes(y, classes=range(1, 7), crop=(224, 224)):
    assert y.ndim == 3 and y.shape[-1] == 2
    if crop is not None:
        y = crop_center(y, crop)
    return tuple(len(np.unique(y[..., 0] * (y[..., 1] == i))) - 1 for i in classes)
