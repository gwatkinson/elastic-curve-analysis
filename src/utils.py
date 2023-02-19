import pickle
import random as rd
from glob import glob
from pathlib import Path
from zipfile import ZipFile

import cv2
import geomstats.backend as gs
import matplotlib.pyplot as plt
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from tqdm.autonotebook import tqdm

M_AMBIENT = 2
k_sampling_points = 500
PRESHAPE_SPACE = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=k_sampling_points)
PRESHAPE_METRIC = PRESHAPE_SPACE.embedding_space.metric


def get_img_contour(img):
    lowerBound = (230, 230, 230)
    upperBound = (255, 255, 255)
    thresh = cv2.inRange(img, lowerBound, upperBound)

    # invert so background black
    thresh = 255 - thresh

    # apply morphology to ensure regions are filled and remove extraneous noise
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    longest_contour = contours[np.argmax([len(_) for _ in contours])]

    return longest_contour


def plot_original_and_contour(img, contour, ax=None):
    ax = ax or plt.gca()
    cv2.drawContours(img, [contour], 0, (36, 255, 12), 3)
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def length_of_contour(contour):
    length = 0
    for i in range(len(contour)):
        length += np.linalg.norm(contour[i] - contour[i - 1]) ** 2
    return np.sqrt(length)


def normalize_contour(contour):
    tmp_contour = list(contour.reshape(-1, 2).copy())
    tmp_contour.append(tmp_contour[0])

    return np.array(tmp_contour) / length_of_contour(tmp_contour)


def exctract_first_n(n_train: int, n_test: int, t: int):
    out_path = Path(f"data/extracts/leaf{t}")
    out_path.mkdir(exist_ok=True, parents=True)
    train_out = out_path / "train"
    train_out.mkdir(exist_ok=True)
    test_out = out_path / "test"
    test_out.mkdir(exist_ok=True)
    zip_file = f"data/zips/leaf{t}.zip"
    files = [(f"l{t}nr{i:0>3}.tif", train_out) for i in range(1, n_train + 1)]
    files += [
        (f"l{t}nr{i:0>3}.tif", test_out)
        for i in range(n_train + 1, n_train + n_test + 1)
    ]

    with ZipFile(zip_file, "r") as zipObj:
        for file, outdir in files:
            try:
                zipObj.extract(file, path=outdir)
            except FileNotFoundError:
                pass


def extract_all_types(n_train: int, n_test: int, n_types: int = 16):
    for t in tqdm(range(1, n_types)):
        exctract_first_n(n_train, n_test, t)
    print("Extracted files from zip")


def generate_ds_from_folders(save=False):
    train_ds = []
    test_ds = []
    for t in tqdm(range(1, 16)):
        train_files = glob(f"data/extracts/leaf{t}/train/*.tif")
        for path in train_files:
            img = cv2.imread(path)
            curve = normalize_contour(get_img_contour(img))
            tmp = {
                "type": t,
                "curve": curve,
                "path": path,
            }
            train_ds.append(tmp)

        test_files = glob(f"data/extracts/leaf{t}/test/*.tif")
        for path in test_files:
            img = cv2.imread(path)
            curve = normalize_contour(get_img_contour(img))
            tmp = {
                "type": t,
                "curve": curve,
                "path": path,
            }
            test_ds.append(tmp)

    rd.shuffle(train_ds)
    rd.shuffle(test_ds)

    if save:
        save_train_test(train_ds, test_ds, folder="data/raw")

    return train_ds, test_ds


def save_train_test(train_ds, test_ds, folder):
    save_out = Path(folder)
    save_out.mkdir(exist_ok=True, parents=True)
    with open(f"{folder}/train_ds.pkl", "wb") as f:
        pickle.dump(train_ds, f)
    with open(f"{folder}/test_ds.pkl", "wb") as f:
        pickle.dump(test_ds, f)
    print("Saved train and test datasets")


def load_train_test(folder):
    save_out = Path(folder)
    save_out.mkdir(exist_ok=True, parents=True)
    with open(f"{folder}/train_ds.pkl", "rb") as f:
        train_ds = pickle.load(f)
    with open(f"{folder}/test_ds.pkl", "rb") as f:
        test_ds = pickle.load(f)
    print("Loaded train and test datasets")

    return train_ds, test_ds


def apply_func_to_ds(input_ds, func):
    """Apply the input function func to the input dictionnary input_ds.

    This function goes through the dictionnary structure and applies
    func to every cell in input_ds[treatment][line].

    It stores the result in a dictionnary output_ds that is returned
    to the user.

    Parameters
    ----------
    input_ds : dict
        Input dictionnary, with keys treatment-line.
    func : callable
        Function to be applied to the values of the dictionnary, i.e.
        the cells.

    Returns
    -------
    output_ds : dict
        Output dictionnary, with the same keys as input_ds.
    """
    output_ds = []
    for plant in input_ds:
        cplant = plant.copy()
        cplant["curve"] = func(plant["curve"])
        output_ds.append(cplant)
    return output_ds


def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return interpolation


def preprocess(curve, tol=1e-10):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """

    dist = curve[1:] - curve[:-1]
    dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

    if np.any(dist_norm < tol):
        for i in range(len(curve) - 1):
            if np.sqrt(np.sum(np.square(curve[i + 1] - curve[i]), axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


def exhaustive_align(curve, base_curve):
    """Align curve to base_curve to minimize the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = PRESHAPE_METRIC.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def preprocess_train_test(train_ds, test_ds, save=False):
    outs = []
    for ds, t in zip([train_ds, test_ds], ["train", "test"]):
        print("Interpolate...")
        ds_interp = apply_func_to_ds(
            input_ds=ds, func=lambda x: interpolate(x, k_sampling_points)
        )
        print("Remove duplicates...")
        ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))
        print("Align...")
        ds_proj = apply_func_to_ds(ds_proc, func=PRESHAPE_SPACE.projection)
        if t == "train":
            BASE_CURVE = ds_proj[0]["curve"]
        ds_align = apply_func_to_ds(
            ds_proj, func=lambda x: exhaustive_align(x, BASE_CURVE)
        )

        outs.append(ds_align)

    train_ds_out, test_ds_out = outs

    if save:
        save_train_test(train_ds_out, test_ds_out, folder="data/processed")

    return train_ds_out, test_ds_out
