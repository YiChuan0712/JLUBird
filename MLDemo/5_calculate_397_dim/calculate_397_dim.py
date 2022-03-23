import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader

from resnest.torch import resnest50

from matplotlib import pyplot as plt

import os, random, gc
import re, time, json
from ast import literal_eval

from IPython.display import Audio
from sklearn.metrics import label_ranking_average_precision_score

from tqdm import tqdm
import joblib

import timm
from sklearn.model_selection import StratifiedGroupKFold

from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import resnest.torch as resnest_torch

""""""
""""""


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

""""""
""""""

NUM_CLASSES = 397
SR = 32_000
DURATION = 7

MAX_READ_SAMPLES = 15 # Each record will have 10 melspecs at most, you can increase this on Colab with High Memory Enabled

""""""
""""""


class Config:
    def __init__(self, debug: bool):
        self.debug = debug

        self.epochs = 1 if self.debug else 50

        self.max_distance = None  # choose from [10, 20, None]
        if self.max_distance is not None:
            self.sites = ["SSW"]  # choose multiples from ["COL", "COR", "SNE", "SSW"]
        else:
            self.sites = None
        self.max_duration = None  # choose from [15, 30, 60, None]
        self.min_rating = None  # choose from [3, 4, None], best: 3?
        self.max_spieces = None  # choose from [100, 200, 300, None], best: 300?
        self.confidence_ub = 0.995  # Probability of birdsong occurrence, default: 0.995, choose from [0.5, 0.7, 0.9, 0.995]
        self.use_high_confidence_only = False  # Whether to use only frames that are likely to be ringing (False performed better).
        self.use_mixup = True
        self.mixup_alpha = 5.0  # 0.5
        self.grouped_by_author = True
        # self.folds = [4]

        self.suffix = f"sr{SR}_d{DURATION}"
        if self.max_spieces:
            self.suffix += f"_spices-{self.max_spieces}"
        if self.min_rating:
            self.suffix += f"_rating-{self.min_rating}"
        if self.use_high_confidence_only:
            self.suffix += f"_high-confidence-only"
        if self.use_mixup:
            self.suffix += f"_miixup-{self.mixup_alpha}"
        if self.grouped_by_author:
            self.suffix += f"_grouped-by-auther"

    def to_dict(self):
        return {
            "debug": self.debug,
            "epochs": self.epochs,
            "max_distance": self.max_distance,
            "sites": self.sites,
            "max_duration": self.max_duration,
            "min_rating": self.min_rating,
            "max_spieces": self.max_spieces,
            "confidence_ub": self.confidence_ub,
            "use_high_confidence_only": self.use_high_confidence_only,
            "use_mixup": self.use_mixup,
            "mixup_alpha": self.mixup_alpha,
            "suffix": self.suffix,
            "grouped_by_author": self.grouped_by_author
        }


config = Config(debug=False)
from pprint import pprint

pprint(config.to_dict())

""""""
""""""

MODEL_NAMES = [
    # "resnext101_32x8d_wsl",
    # 'efficientnet_b0',
    "resnest50",
    # "densenet121",
]

""""""
""""""

MEL_PATHS = sorted(Path("../input").glob("kkiller-birdclef-mels-computer-d7-part/rich_train_metadata.csv"))
TRAIN_LABEL_PATHS = sorted(Path("../input").glob("kkiller-birdclef-mels-computer-d7-part/LABEL_IDS.json"))

MODEL_ROOT = Path(".")

""""""
""""""

TRAIN_BATCH_SIZE = 50 # 16
TRAIN_NUM_WORKERS = 0

VAL_BATCH_SIZE = 50 # 16 # 128
VAL_NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

""""""
""""""

checkpoint_paths = [
    # model1
    Path("../input/clefmodel/birdclef_resnest50_fold0_epoch_27_f1_val_05179_20210520120053.pth"),
]

""""""
""""""


def get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS):
    df = None
    LABEL_IDS = {}

    for file_path in mel_paths:
        temp = pd.read_csv(str(file_path), index_col=0)
        temp["impath"] = temp.apply(
            lambda row: file_path.parent / "audio_images/{}/{}.npy".format(row.primary_label, row.filename), axis=1)
        df = temp if df is None else df.append(temp)

    df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)

    for file_path in train_label_paths:
        with open(str(file_path)) as f:
            LABEL_IDS.update(json.load(f))

    return LABEL_IDS, df

""""""
""""""

from typing import List
def get_locations() -> List[dict]:
    return [{
        "site": "COL",
        "latitude": 5.57,
        "longitude": -75.85
    }, {
        "site": "COR",
        "latitude": 10.12,
        "longitude": -84.51
    }, {
        "site": "SNE",
        "latitude": 38.49,
        "longitude": -119.95
    }, {
        "site": "SSW",
        "latitude": 42.47,
        "longitude": -76.45
    }]

def is_in_site(row, sites, max_distance):
    for location in get_locations():
        if location["site"] in sites:
            x = (row["latitude"] - location["latitude"])
            y = (row["longitude"] - location["longitude"])
            r = (x**2 + y**2) ** 0.5
            if r < max_distance:
                return True
    return False

""""""
""""""

LABEL_IDS, df = get_df()

if config.grouped_by_author:
    kf = StratifiedGroupKFold(n_splits=5)
    x = df[["latitude", "longitude"]].values
    y = df["label_id"].values
    groups = df["author"].values
    df["fold"] = -1
    for kfold_index, (train_index, valid_index) in enumerate(kf.split(x, y, groups)):
        df.loc[valid_index, "fold"] = kfold_index

if config.debug:
    df = df.head(100)

print("before:%d" % len(df))
# Within a certain distance of the target area
if config.max_distance is not None:
    df = df[df.apply(lambda row: is_in_site(row, config.sites, config.max_distance), axis=1)]
# Number of Species
if config.max_spieces is not None:
    s = df["primary_label"].value_counts().head(config.max_spieces)
    df = df[df["primary_label"].isin(s.index)]
# Rating is above a certain value
if config.min_rating is not None:
    df = df[df["rating"] >= config.min_rating]
# Within a certain amount of recording time
if config.max_duration is not None:
    df = df[df["duration"] < config.max_duration]
df = df.reset_index(drop=True)
print("after:%d" % len(df))

print(df.shape)
df.head()

""""""
""""""


def get_model(name, num_classes=NUM_CLASSES):
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if "resnest" in name:
        # if not os.path.exists("resnest50-528c19ca.pth"):
        #     !wget
        #     https: // github.com / rwightman / pytorch - image - models / releases / download / v0
        #     .1 - resnest / resnest50 - 528
        #     c19ca.pth

        pretrained_weights = torch.load('resnest50-528c19ca.pth')
        model = getattr(resnest_torch, name)(pretrained=False)
        model.load_state_dict(pretrained_weights)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif name.startswith("efficientnet_b"):
        model = getattr(timm.models.efficientnet, name)(pretrained=True)
    elif name.startswith("densenet"):
        model = getattr(timm.models.densenet, name)(pretrained=True)
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model

""""""
""""""


class BirdClefDataset(Dataset):
    def __init__(
            self,
            meta,
            sr=SR,
            is_train=True,
            num_classes=NUM_CLASSES,
            duration=DURATION
    ):
        self.meta = meta.copy().reset_index(drop=True)
        records = []
        for idx, row in tqdm(self.meta.iterrows(), total=len(self.meta)):
            images = np.load(str(row["impath"]))
            for i, image in enumerate(images):
                seconds = i * duration
                records.append({
                    "filename": row["filename"],
                    "impath": row["impath"],
                    "seconds": seconds,
                    "index": i
                })
        self.records = records
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.eps = 0.0025

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image = np.load(str(row["impath"]))[row["index"]]
        image = self.normalize(image)
        return image, row["filename"], row["seconds"]

""""""
""""""

nocall_df = pd.read_csv("../input/train_short_audio_nocall_fold0to4/nocalldetection_for_shortaudio_fold0.csv")

""""""
""""""

ds = BirdClefDataset(meta=df, sr=SR, duration=DURATION, is_train=True)
print(len(df))

""""""
""""""

def add_tail(model, num_classes):
    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)
    return model

def load_net(checkpoint_path, num_classes=NUM_CLASSES):
    if "resnest50" in checkpoint_path:
        net = resnest50(pretrained=False)
    elif "resnest26d" in checkpoint_path:
        net = timm.models.resnest26d(pretrained=False)
    elif "resnext101_32x8d_wsl" in checkpoint_path:
        net = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    elif "efficientnet_b0" in checkpoint_path:
        net = getattr(timm.models.efficientnet, "efficientnet_b0")(pretrained=False)
    elif "densenet121" in checkpoint_path:
        net = timm.models.densenet121(pretrained=False)
    else:
        raise ValueError("Unexpected checkpont name: %s" % checkpoint_path)
    net = add_tail(net, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

""""""
""""""

@torch.no_grad()
def predict(net, criterion, val_laoder):
    net.eval()
    records = []
    val_laoder = tqdm(val_laoder, leave = False, total=len(val_laoder))
    for icount, (xb, filename, seconds) in enumerate(val_laoder):
        xb = xb.to(DEVICE)
        prob = net(xb)
        prob = torch.sigmoid(prob)
        records.append({
            "prob": prob,
            "filename": filename,
            "seconds": seconds
        })
    return records

""""""
""""""

def one_fold(checkpoint_path, fold, train_set, val_set, epochs=20, save=True, save_root=None):
    net = load_net(checkpoint_path)
    criterion = nn.BCEWithLogitsLoss()
    val_data = BirdClefDataset(
        meta=df.iloc[val_set].reset_index(drop=True),
        sr=SR,
        duration=DURATION,
        is_train=False
    )
    val_laoder = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)
    y_preda = predict(net, criterion, val_laoder)
    return y_preda

""""""
""""""


def predict_for_oof(checkpoint_path, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
    gc.collect()
    torch.cuda.empty_cache()

    fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max() + 1)

    for fold, val_set in fold_bar:
        if folds and not fold in folds:
            continue

        print(f"\n############################### [FOLD {fold}]")
        fold_bar.set_description(f"[FOLD {fold}]")
        train_set = np.setdiff1d(df.index, val_set)
        records = one_fold(checkpoint_path, fold=fold, train_set=train_set, val_set=val_set, epochs=epochs, save=save,
                           save_root=save_root)
        gc.collect()
        torch.cuda.empty_cache()
        return records

""""""
""""""

def to_call_prob(row):
    i = row["seconds"] // DURATION
    call_prob = float(row["nocalldetection"].split()[i])
    return call_prob

""""""
""""""

def to_birds(row):
    if row["call_prob"] < 0.5:
        return "nocall"
    res = [row["primary_label"]] + eval(row["secondary_labels"])
    return " ".join(res)

""""""
""""""

INV_LABEL_IDS = {v:k for k, v in LABEL_IDS.items()}
columns = [INV_LABEL_IDS[i] for i in range(len(LABEL_IDS))]
metadata_df = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
nocall_df = pd.read_csv("../input/train_short_audio_nocall_fold0to4/nocalldetection_for_shortaudio_fold0.csv")

""""""
""""""

import glob
filename_to_nocalldetection = {}
filepath_list = list(glob.glob("../input/train_short_audio_nocall_fold0to4/*.csv"))
for filepath in filepath_list:
    nocall_df = pd.read_csv(filepath)
    probs = nocall_df["nocalldetection"].apply(
        lambda _: list(
            map(float, _.split())
        )
    ).tolist()
    for k, v in zip(nocall_df["filename"].tolist(), probs):
        if not k in filename_to_nocalldetection:
            filename_to_nocalldetection[k] = v
        else:
            w = filename_to_nocalldetection[k]
            for i in range( len(w)):
                w[i] += v[i]
            filename_to_nocalldetection[k] = w

for k, v in filename_to_nocalldetection.items():
    for i in range(len(v)):
        filename_to_nocalldetection[k][i] /= len(filepath_list)

for k, v in filename_to_nocalldetection.items():
    filename_to_nocalldetection[k] = " ".join(map(str, v))

""""""
""""""

nocall_df = pd.DataFrame(filename_to_nocalldetection.items(), columns=["filename", "nocalldetection"])

""""""
""""""

nocall_df.head()

""""""
""""""

filepath_list = []
for checkpoint_path in checkpoint_paths:
    print("\n\n###########################################", checkpoint_path)
    # Find out which fold it is from the name of the model file.
    fold = -1
    for i in range(5):
        if f"fold{i}" in checkpoint_path.stem:
            fold = i
            break
    print("target validation fold is %d" % fold)
    if fold == -1:
        raise ValueError("Unexpected fold value")
    # Run on the fold that is the target of oof.
    records_list = predict_for_oof(checkpoint_path.as_posix(), epochs=config.epochs, suffix=config.suffix, folds=[fold])
    dfs = []
    for records in records_list:
        prob = records["prob"].to("cpu").numpy()
        _df = pd.DataFrame(prob)
        _df.columns = columns
        _df["seconds"] = records["seconds"].to("cpu").numpy().tolist()
        _df["filename"] = list(records["filename"])
        dfs.append(_df)
    oof_df = pd.concat(dfs)
    oof_df = pd.merge(oof_df, metadata_df, how="left", on=["filename"])
    oof_df = pd.merge(oof_df, nocall_df[["filename", "nocalldetection"]], how="left", on=["filename"])
    oof_df["call_prob"] = oof_df.apply(to_call_prob, axis=1)
    oof_df["birds"] = oof_df.apply(to_birds, axis=1)
    filepath = "%s.csv" % checkpoint_path.stem
    print(f"Save to {filepath}")
    oof_df.drop(
        columns=[
            'scientific_name',
            'common_name',
            'license',
            'time',
            'url',
            'nocalldetection',
        ]
    ).to_csv(filepath, index=False)
    filepath_list.append(filepath)
