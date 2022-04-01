import warnings
warnings.filterwarnings('ignore')

""""""
""""""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import re
import time

import pickle
from typing import List
from tqdm import tqdm

# sound
import librosa as lb
import soundfile as sf

# pytorch
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50

import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import pickle
from catboost import CatBoostClassifier
from catboost import Pool
# from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import random


def classification(TEST_AUDIO_ROOT, rid, rlocation, rdate, mnum):

    BIRD_LIST = sorted(os.listdir('D:\\birdclefFirstPlace/input/birdclef-2021/train_short_audio'))
    BIRD2IDX = {bird:idx for idx, bird in enumerate(BIRD_LIST)}
    BIRD2IDX['nocall'] = -1
    IDX2BIRD = {idx:bird for bird, idx in BIRD2IDX.items()}

    class TrainingConfig:
        def __init__(self):
            self.nocall_threshold: float = 0.5
            self.num_kfolds: int = 5
            self.num_spieces: int = 397
            self.num_candidates: int = 5
            self.max_distance: int = 15  # 20
            self.sampling_strategy: float = None  # 1.0
            self.random_state: int = 777
            self.num_prob: int = 6
            self.use_to_birds = True
            self.weights_filepath_dict = {
                'lgbm': [f"D:\\birdclefFirstPlace/6_step_three/lgbm_{kfold_index}.pkl" for kfold_index in range(self.num_kfolds)],
            }


    training_config = TrainingConfig()

    """"""
    """"""


    class Config:
        def __init__(self, mn:int):
            self.num_kfolds: int = training_config.num_kfolds
            self.num_spieces: int = training_config.num_spieces
            self.num_candidates: int = training_config.num_candidates
            self.max_distance: int = training_config.max_distance
            self.nocall_threshold: float = training_config.nocall_threshold
            self.num_prob: int = training_config.num_prob
            # check F1 score without 3rd stage(table competition)
            self.check_baseline: bool = True
            # List of file paths of the models which are used when determining if the bird is acceptable.
            self.weights_filepath_dict = training_config.weights_filepath_dict
            # Weights for the models to predict the probability of each bird singing for each frame.
            self.checkpoint_paths = [
                Path("D:\\birdclefFirstPlace/input/clefmodel/birdclef_resnest50_fold0_epoch_27_f1_val_05179_20210520120053.pth"),  # id36
                Path("D:\\birdclefFirstPlace/input/clefmodel/birdclef_resnest50_fold0_epoch_13_f1_val_03502_20210522050604.pth"),  # id51
                Path("D:\\birdclefFirstPlace/input/birdclef_groupby_author_05221040_728258/birdclef_resnest50_fold0_epoch_33_f1_val_03859_20210524151554.pth"),
                # id58
                Path("D:\\birdclefFirstPlace/input/birdclef_groupby_author_05221040_728258/birdclef_resnest50_fold1_epoch_34_f1_val_04757_20210524185455.pth"),
                # id59
                Path("D:\\birdclefFirstPlace/input/birdclef_groupby_author_05221040_728258/birdclef_resnest50_fold2_epoch_34_f1_val_05027_20210524223209.pth"),
                # id60
                Path("D:\\birdclefFirstPlace/input/birdclef_groupby_author_05221040_728258/birdclef_resnest50_fold3_epoch_20_f1_val_04299_20210525010703.pth"),
                # id61
                Path("D:\\birdclefFirstPlace/input/birdclef_groupby_author_05221040_728258/birdclef_resnest50_fold4_epoch_34_f1_val_05140_20210525074929.pth"),
                # id62
                Path("D:\\birdclefFirstPlace/input/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold0_epoch_78_f1_val_03658_20210528221629.pth"),
                # id97
                Path("D:\\birdclefFirstPlace/input/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold0_epoch_84_f1_val_03689_20210528225810.pth"),
                # id97
                Path("D:\\birdclefFirstPlace/input/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold1_epoch_27_f1_val_03942_20210529062427.pth"),
                # id98
            ]
            self.checkpoint_paths = self.checkpoint_paths[:mn]
            # call probability of each bird for each sample used for candidate extraction (cache)
            self.pred_filepath_list = [
                self.get_prob_filepath_from_checkpoint(path) for path in self.checkpoint_paths
            ]

        def get_prob_filepath_from_checkpoint(self, checkpoint_path: Path) -> str:
            filename = f"train_soundscape_labels_probabilitiy_%s.csv" % checkpoint_path.stem
            return filename


    config = Config(mnum)

    """"""
    """"""

    def get_locations():
        # 音频来源 一共有四个地点
        # 以及对应地点的经纬度
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


    def to_site(row, max_distance: int):
        # 给出一行 写入对应的地点
        best = max_distance  # 距离阈值 可以人工控制
        answer = "Other"  # 默认地点为other
        # 遍历每一个地点
        for location in get_locations():
            x = (row["latitude"] - location["latitude"])
            y = (row["longitude"] - location["longitude"])
            dist = (x**2 + y**2) ** 0.5
            if dist < best:
                best = dist
                answer = location["site"]
        return answer


    def to_latitude(site: str) -> str:
        # 给出site 返回latitude
        for location in get_locations():
            if site == location["site"]:
                return location["latitude"]
        return -10000


    def to_longitude(site: str) -> str:
        # 给出site 返回longitude
        for location in get_locations():
            if site == location["site"]:
                return location["longitude"]
        return -10000


    def to_birds(row, th: float) -> str:
        if row["call_prob"] < th:
            return "nocall"
        res = [row["primary_label"]] + eval(row["secondary_labels"])
        return " ".join(res)


    def make_candidates(
            prob_df: pd.DataFrame,
            num_spieces: int,
            num_candidates: int,
            max_distance: int,
            num_prob: int = 6,  # number of frames to be allocated for front and rear (if 3, then 3 for front, 3 for rear)
            nocall_threshold: float = 0.5,
    ):
        if "author" in prob_df.columns:  # meta data (train_short_audio)
            print("has author")
            prob_df["birds"] = prob_df.apply(
                lambda row: to_birds(row, th=nocall_threshold),
                axis=1
            )
            print("Candidate nocall ratio: %.4f" % (prob_df["birds"] == "nocall").mean())
            prob_df["audio_id"] = prob_df["filename"].apply(
                lambda _: int(_.replace("XC", "").replace(".ogg", ""))
            )
            prob_df["row_id"] = prob_df.apply(
                lambda row: "%s_%s" % (row["audio_id"], row["seconds"]),
                axis=1
            )
            prob_df["year"] = prob_df["date"].apply(lambda _: int(_.split("-")[0]))
            prob_df["month"] = prob_df["date"].apply(lambda _: int(_.split("-")[1]))
            prob_df["site"] = prob_df.apply(
                lambda row: to_site(row, max_distance),
                axis=1
            )
        else:
            print("no author")
            prob_df["year"] = prob_df["date"].apply(lambda _: int(str(_)[:4]))
            prob_df["month"] = prob_df["date"].apply(lambda _: int(str(_)[4:6]))
            prob_df["latitude"] = prob_df["site"].apply(to_latitude)
            prob_df["longitude"] = prob_df["site"].apply(to_longitude)

        sum_prob_list = prob_df[BIRD_LIST].sum(axis=1).tolist()
        mean_prob_list = prob_df[BIRD_LIST].mean(axis=1).tolist()
        std_prob_list = prob_df[BIRD_LIST].std(axis=1).tolist()
        max_prob_list = prob_df[BIRD_LIST].max(axis=1).tolist()
        min_prob_list = prob_df[BIRD_LIST].min(axis=1).tolist()
        skew_prob_list = prob_df[BIRD_LIST].skew(axis=1).tolist()
        kurt_prob_list = prob_df[BIRD_LIST].kurt(axis=1).tolist()

        X = prob_df[BIRD_LIST].values
        bird_ids_list = np.argsort(-X)[:, :num_candidates]
        row_ids = prob_df["row_id"].tolist()
        rows = [i // num_candidates for i in range(len(bird_ids_list.flatten()))]
        cols = bird_ids_list.flatten()
        # What number?
        ranks = [i % num_candidates for i in range(len(rows))]
        probs_list = X[rows, cols]
        D = {
            "row_id": [row_ids[i] for i in rows],
            "rank": ranks,
            "bird_id": bird_ids_list.flatten(),
            "prob": probs_list.flatten(),
            "sum_prob": [sum_prob_list[i // num_candidates] for i in range(num_candidates * len(mean_prob_list))],
            "mean_prob": [mean_prob_list[i // num_candidates] for i in range(num_candidates * len(mean_prob_list))],
            "std_prob": [std_prob_list[i // num_candidates] for i in range(num_candidates * len(std_prob_list))],
            "max_prob": [max_prob_list[i // num_candidates] for i in range(num_candidates * len(max_prob_list))],
            "min_prob": [min_prob_list[i // num_candidates] for i in range(num_candidates * len(min_prob_list))],
            "skew_prob": [skew_prob_list[i // num_candidates] for i in range(num_candidates * len(skew_prob_list))],
            "kurt_prob": [kurt_prob_list[i // num_candidates] for i in range(num_candidates * len(kurt_prob_list))],
        }
        audio_ids = prob_df["audio_id"].values[rows]
        for diff in range(-num_prob, num_prob + 1):
            if diff == 0:
                continue
            neighbor_audio_ids = prob_df["audio_id"].shift(diff).values[rows]
            Y = prob_df[BIRD_LIST].shift(diff).values
            c = f"next{abs(diff)}_prob" if diff < 0 else f"prev{diff}_prob"
            c = c.replace("1_prob", "_prob")  # Fix next1_prob to next_prob
            v = Y[rows, cols].flatten()
            v[audio_ids != neighbor_audio_ids] = np.nan
            D[c] = v

        candidate_df = pd.DataFrame(D)
        columns = [
            "row_id",
            "site",
            "year",
            "month",
            "audio_id",
            "seconds",
            "birds",
        ]
        candidate_df = pd.merge(
            candidate_df,
            prob_df[columns],
            how="left",
            on="row_id"
        )

        print(candidate_df)
        candidate_df.info()

        candidate_df["target"] = candidate_df.apply(
            lambda row: IDX2BIRD[row["bird_id"]] in set(row["birds"].split()),
            axis=1
        )
        candidate_df["label"] = candidate_df["bird_id"].map(IDX2BIRD)
        return candidate_df

    """"""
    """"""

    def load_metadata():
        meta_df = pd.read_csv("D:\\birdclefFirstPlace/input/birdclef-2021/train_metadata.csv")
        meta_df["id"] = meta_df.index + 1
        meta_df["year"] = meta_df["date"].apply(lambda _: _.split("-")[0]).astype(int)
        meta_df["month"] = meta_df["date"].apply(lambda _: _.split("-")[1]).astype(int)
        return meta_df


    def to_zscore(row):
        x = row["prob"]
        mu = row["prob_avg_in_same_audio"]
        sigma = row["prob_var_in_same_audio"] ** 0.5
        if sigma < 1e-6:
            return 0
        else:
            return (x - mu) / sigma


    def add_same_audio_features(
        candidate_df:pd.DataFrame,
        df:pd.DataFrame
    ):
        # Average probability per bird in the same audio
        _gdf = df.groupby(["audio_id"], as_index=False).mean()[["audio_id"] + BIRD_LIST]
        _df = pd.melt(
            _gdf,
            id_vars=["audio_id"]
        ).rename(columns={
            "variable": "label",
            "value": "prob_avg_in_same_audio"
        })
        candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
        # Maximum value for each bird in the same audio
        _gdf = df.groupby(["audio_id"], as_index=False).max()[["audio_id"] + BIRD_LIST]
        _df = pd.melt(
            _gdf,
            id_vars=["audio_id"]
        ).rename(columns={
            "variable": "label",
            "value": "prob_max_in_same_audio"
        })
        candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
        # Variance of each bird in the same audio
        _gdf = df.groupby(["audio_id"], as_index=False).var()[["audio_id"] + BIRD_LIST]
        _df = pd.melt(
            _gdf,
            id_vars=["audio_id"]
        ).rename(columns={
            "variable": "label",
            "value": "prob_var_in_same_audio"
        })
        candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
        candidate_df["zscore_in_same_audio"] = candidate_df.apply(to_zscore, axis=1)
        return candidate_df

    """"""
    """"""

    def add_features(
        candidate_df:pd.DataFrame,
        df:pd.DataFrame,
        max_distance:int,
    ):
        meta_df = load_metadata()
        # latitude & longitude
        if not "latitude" in candidate_df.columns:
            candidate_df["latitude"] = candidate_df["site"].apply(to_latitude)
        if not "longitude" in candidate_df.columns:
            candidate_df["longitude"] = candidate_df["site"].apply(to_longitude)
        # Number of Appearances
        candidate_df["num_appear"] = candidate_df["label"].map(
            meta_df["primary_label"].value_counts()
        )
        meta_df["site"] = meta_df.apply(
            lambda row: to_site(
                row,
                max_distance=max_distance
            ),
            axis=1
        )

        # Number of occurrences by region
        _df = meta_df.groupby(
            ["primary_label", "site"],
            as_index=False
        )["id"].count().rename(
            columns={
                "primary_label": "label",
                "id": "site_num_appear"
            }
        )
        candidate_df = pd.merge(
            candidate_df,
            _df,
            how="left",
            on=["label", "site"]
        )
        candidate_df["site_appear_ratio"] = candidate_df["site_num_appear"] / candidate_df["num_appear"]
        # Seasonal statistics
        _df = meta_df.groupby(
            ["primary_label", "month"],
            as_index=False
        )["id"].count().rename(
            columns={
                "primary_label": "label",
                "id": "month_num_appear"
            }
        )
        candidate_df = pd.merge(candidate_df, _df, how="left", on=["label", "month"])
        candidate_df["month_appear_ratio"] = candidate_df["month_num_appear"] / candidate_df["num_appear"]

        candidate_df = add_same_audio_features(candidate_df, df)

        # Correction of probability (all down)
        candidate_df["prob / num_appear"] = candidate_df["prob"] / (candidate_df["num_appear"].fillna(0) + 1)
        candidate_df["prob / site_num_appear"] = candidate_df["prob"] / (candidate_df["site_num_appear"].fillna(0) + 1)
        candidate_df["prob * site_appear_ratio"] = candidate_df["prob"] * (candidate_df["site_appear_ratio"].fillna(0) + 0.001)

        # Amount of change from the previous and following frames
        candidate_df["prob_avg"] = candidate_df[["prev_prob", "prob", "next_prob"]].mean(axis=1)
        candidate_df["prob_diff"] = candidate_df["prob"] - candidate_df["prob_avg"]
        candidate_df["prob - prob_max_in_same_audio"] = candidate_df["prob"] - candidate_df["prob_max_in_same_audio"]

        # Average of back and forward frames

        return candidate_df

    """"""
    """"""

    # DEVICE = torch.device("cuda")
    DEVICE = torch.device("cpu")
    print("DEVICE:", DEVICE)


    class MelSpecComputer:
        def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
            self.sr = sr
            self.n_mels = n_mels
            self.fmin = fmin
            self.fmax = fmax
            kwargs["n_fft"] = kwargs.get("n_fft", self.sr // 10)
            kwargs["hop_length"] = kwargs.get("hop_length", self.sr // (10 * 4))
            self.kwargs = kwargs

        def __call__(self, y):
            melspec = lb.feature.melspectrogram(
                y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
            )

            melspec = lb.power_to_db(melspec).astype(np.float32)
            return melspec


    def mono_to_color(X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V


    def crop_or_pad(y, length):
        if len(y) < length:
            y = np.concatenate([y, length - np.zeros(len(y))])
        elif len(y) > length:
            y = y[:length]
        return y


    class BirdCLEFDataset(Dataset):
        def __init__(self, data, sr=32_000, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast",
                     resample=True):

            self.data = data

            self.sr = sr
            self.n_mels = n_mels
            self.fmin = fmin
            self.fmax = fmax or self.sr // 2

            self.duration = duration
            self.audio_length = self.duration * self.sr
            self.step = step or self.audio_length

            self.res_type = res_type
            self.resample = resample

            self.mel_spec_computer = MelSpecComputer(
                sr=self.sr,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            self.npy_save_root = Path("D:\\birdclefFirstPlace/6_step_three/data")

            os.makedirs(self.npy_save_root, exist_ok=True)

        def __len__(self):
            return len(self.data)

        @staticmethod
        def normalize(image):
            image = image.astype("float32", copy=False) / 255.0
            image = np.stack([image, image, image])
            return image

        def audio_to_image(self, audio):
            melspec = self.mel_spec_computer(audio)
            image = mono_to_color(melspec)
            image = self.normalize(image)
            return image

        def read_file(self, filepath):
            filename = Path(filepath).stem
            npy_path = self.npy_save_root / f"{filename}.npy"

            if not os.path.exists(npy_path):
                audio, orig_sr = sf.read(filepath, dtype="float32")

                if self.resample and orig_sr != self.sr:
                    audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

                audios = []
                for i in range(self.audio_length, len(audio) + self.step, self.step):
                    start = max(0, i - self.audio_length)
                    end = start + self.audio_length
                    audios.append(audio[start:end])

                if len(audios[-1]) < self.audio_length:
                    audios = audios[:-1]

                images = [self.audio_to_image(audio) for audio in audios]
                images = np.stack(images)

                np.save(str(npy_path), images)
            return np.load(npy_path)

        def __getitem__(self, idx):
            return self.read_file(self.data.loc[idx, "filepath"])


    def load_net(checkpoint_path, num_classes=397):
        net = resnest50(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        dummy_device = torch.device("cpu")
        d = torch.load(checkpoint_path, map_location=dummy_device)
        for key in list(d.keys()):
            d[key.replace("model.", "")] = d.pop(key)
        net.load_state_dict(d)
        net = net.to(DEVICE)
        net = net.eval()
        return net


    @torch.no_grad()
    def get_thresh_preds(out, thresh=None):
        thresh = thresh or THRESH
        o = (-out).argsort(1)
        npreds = (out > thresh).sum(1)
        preds = []
        for oo, npred in zip(o, npreds):
            preds.append(oo[:npred].cpu().numpy().tolist())
        return preds


    def predict(nets, test_data, names=True):
        preds = []
        with torch.no_grad():
            for idx in tqdm(list(range(len(test_data)))):
                xb = torch.from_numpy(test_data[idx]).to(DEVICE)
                pred = 0.
                for net in nets:
                    o = net(xb)
                    o = torch.sigmoid(o)
                    pred += o
                pred /= len(nets)
                if names:
                    pred = BIRD_LIST(get_thresh_preds(pred))

                preds.append(pred)
        return preds

    """"""
    """"""

    def get_prob_df(config, audio_paths, rid,  rlocation, rdate):
        data = pd.DataFrame(
             # [(path.stem, *path.stem.split("_"), path) for path in Path(audio_paths).glob("*.ogg")],
            [[Path(audio_paths).stem, rid, rlocation, rdate, audio_paths]],
            columns = ["filename", "id", "site", "date", "filepath"]
        )
        test_data = BirdCLEFDataset(data=data)

        for checkpoint_path in config.checkpoint_paths:
            prob_filepath = config.get_prob_filepath_from_checkpoint(checkpoint_path)
            if (not os.path.exists(prob_filepath)) or (TARGET_PATH is None):  # Always calculate when no cash is available or when submitting.
                nets = [load_net(checkpoint_path.as_posix())]
                pred_probas = predict(nets, test_data, names=False)
                print("a")
                if TARGET_PATH:  # local
                    df = pd.read_csv(TARGET_PATH, usecols=["row_id", "birds"])
                    print("b")
                else:  # when it is submission
                    if str(audio_paths) == "D:\\birdclefFirstPlace/input/birdclef-2021/train_soundscapes":
                        print(audio_paths)
                        df = pd.read_csv(Path("D:\\birdclefFirstPlace/input/birdclef-2021/train_soundscape_labels.csv"), usecols=["row_id", "birds"])
                        print("c")
                    else:
                        print(SAMPLE_SUB_PATH)
                        df = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id", "birds"])
                        print("d")
                df["audio_id"] = df["row_id"].apply(lambda _: int(_.split("_")[0]))
                df["site"] = df["row_id"].apply(lambda _: _.split("_")[1])
                df["seconds"] = df["row_id"].apply(lambda _: int(_.split("_")[2]))
                assert len(data) == len(pred_probas)
                n = len(data)
                audio_id_to_date = {}
                audio_id_to_site = {}
                # for filepath in audio_paths.glob("*.ogg"):
                #     audio_id, site, date = os.path.basename(filepath).replace(".ogg", "").split("_")
                #     audio_id = int(audio_id)
                #     audio_id_to_date[audio_id] = date
                #     audio_id_to_site[audio_id] = site
                audio_id = int(rid)
                audio_id_to_date[audio_id] = rdate
                audio_id_to_site[audio_id] = rlocation
                dfs = []
                for i in range(n):
                    row = data.iloc[i]
                    audio_id = int(row["id"])
                    pred = pred_probas[i]
                    _df = pd.DataFrame(pred.to("cpu").numpy())
                    _df.columns = [IDX2BIRD[j] for j in range(_df.shape[1])]
                    _df["audio_id"] = audio_id
                    _df["date"] = audio_id_to_date[audio_id]
                    _df["site"] = audio_id_to_site[audio_id]
                    _df["seconds"] = df["seconds"]
                    # _df["seconds"] = [(j+1)*5 for j in range(120)]
                    dfs.append(_df)
                prob_df = pd.concat(dfs)
                prob_df = pd.merge(prob_df, df, how="left", on=["site", "audio_id", "seconds"])
                print(f"Save to {prob_filepath}")
                prob_df.to_csv(prob_filepath, index=False)

        # Ensemble
        prob_df = pd.read_csv(
            config.get_prob_filepath_from_checkpoint(config.checkpoint_paths[0])
        )
        if len(config.checkpoint_paths) > 1:
            columns = BIRD_LIST
            for checkpoint_path in config.checkpoint_paths[1:]:
                _df = pd.read_csv(
                    config.get_prob_filepath_from_checkpoint(checkpoint_path)
                )
                prob_df[columns] += _df[columns]
            prob_df[columns] /= len(config.checkpoint_paths)

        return prob_df

    """"""
    """"""

    def seed_everything(seed=1234):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    """"""
    """"""


    def train(
            candidate_df: pd.DataFrame,
            df: pd.DataFrame,
            candidate_df_soundscapes: pd.DataFrame,
            df_soundscapes: pd.DataFrame,
            num_kfolds: int,
            num_candidates: int,
            verbose: bool = False,
            sampling_strategy: float = 1.0,
            random_state: int = 777,
    ):
        seed_everything(random_state)
        feature_names = get_feature_names()

        if verbose:
            print("features", feature_names)

        # short audio の  k fold
        candidate_df.info()
        groups = candidate_df["audio_id"]
        # print(groups)
        # exit(0)
        kf = StratifiedGroupKFold(
            n_splits=num_kfolds)  # When using lgbm_rank, it is necessary to use the data attached to each group, so don't shuffle them.
        for kfold_index, (_, valid_index) in enumerate(
                kf.split(candidate_df[feature_names].values, candidate_df["target"].values, groups)):
            candidate_df.loc[valid_index, "fold"] = kfold_index

        X = candidate_df[feature_names].values
        y = candidate_df["target"].values
        print(y)
        print("-"*100)
        oofa = np.zeros(len(candidate_df_soundscapes), dtype=np.float32)

        for kfold_index in range(num_kfolds):
            print(f"fold {kfold_index}")
            train_index = candidate_df[candidate_df["fold"] != kfold_index].index
            valid_index = candidate_df[candidate_df["fold"] == kfold_index].index
            X_train, y_train = X[train_index], y[train_index]
            # X_valid, y_valid = X[valid_index], y[valid_index]
            X_valid, y_valid = candidate_df_soundscapes[feature_names].values, candidate_df_soundscapes["target"].values

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'device': 'gpu',
            }
            # model = lgb.train(
            #     params,
            #     dtrain,
            #     valid_sets=dvalid,
            #     num_boost_round=200,
            #     early_stopping_rounds=20,
            #     verbose_eval=20,
            # )
            model = pickle.load(open(f"lgbm_{kfold_index}.pkl", "rb"))
            oofa += model.predict(X_valid.astype(np.float32)) / num_kfolds
            # pickle.dump(model, open(f"lgbm_{kfold_index}.pkl", "wb"))
        # for i in range(len(oofa)):
        #     print(oofa[i])
        # exit(0)

        def f(th):
            _df = candidate_df_soundscapes[(oofa > th)]
            if len(_df) == 0:
                return 0
            _gdf = _df.groupby(
                ["audio_id", "seconds"],
                as_index=False
            )["label"].apply(lambda _: " ".join(_))
            df2 = pd.merge(
                df_soundscapes[["audio_id", "seconds", "birds"]],
                _gdf,
                how="left",
                on=["audio_id", "seconds"]
            )
            df2.loc[df2["label"].isnull(), "label"] = "nocall"
            return df2.apply(
                lambda _: get_metrics(_["birds"], _["label"])["f1"],
                axis=1
            ).mean()

        print("-" * 30)
        print(f"#sound_scapes (len:{len(candidate_df_soundscapes)}) でのスコア")
        lb, ub = 0, 1
        for k in range(30):
            th1 = (2 * lb + ub) / 3
            th2 = (lb + 2 * ub) / 3
            if f(th1) < f(th2):
                lb = th1
            else:
                ub = th2
        th = (lb + ub) / 2
        print("best th: %.4f" % th)
        print("best F1: %.4f" % f(th))
        if verbose:
            y_soundscapes = candidate_df_soundscapes["target"].values
            oof = (oofa > th).astype(int)
            print("[details] Call or No call classification")
            print("binary F1: %.4f" % f1_score(y_soundscapes, oof))
            print("gt positive ratio: %.4f" % np.mean(y_soundscapes))
            print("oof positive ratio: %.4f" % np.mean(oof))
            print("Accuracy: %.4f" % accuracy_score(y_soundscapes, oof))
            print("Recall: %.4f" % recall_score(y_soundscapes, oof))
            print("Precision: %.4f" % precision_score(y_soundscapes, oof))
        print("-" * 30)
        print()

    """"""
    """"""

    def get_feature_names() -> List[str]:
        return [
            "year",
            "month",
            "sum_prob",
            "mean_prob",
            #"std_prob",
            "max_prob",
            #"min_prob",
            #"skew_prob",
            #"kurt_prob",
            "prev6_prob",
            "prev5_prob",
            "prev4_prob",
            "prev3_prob",
            "prev2_prob",
            "prev_prob",
            "prob",
            "next_prob",
            "next2_prob",
            "next3_prob",
            "next4_prob",
            "next5_prob",
            "next6_prob",
            "rank",
            "latitude",
            "longitude",
            "bird_id", # +0.013700
            "seconds", # -0.0050
            "num_appear",
            "site_num_appear",
            "site_appear_ratio",
            # "prob / num_appear", # -0.005
            # "prob / site_num_appear", # -0.0102
            # "prob * site_appear_ratio", # -0.0049
            # "prob_avg", # -0.0155
            "prob_diff", # 0.0082
            # "prob_avg_in_same_audio", # -0.0256
            # "prob_max_in_same_audio", # -0.0142
            # "prob_var_in_same_audio", # -0.0304
            # "prob - prob_max_in_same_audio", # -0.0069
            # "zscore_in_same_audio", # -0.0110
            # "month_num_appear", # 0.0164
        ]


    def get_metrics(s_true, s_pred):
        s_true = set(s_true.split())
        s_pred = set(s_pred.split())
        n, n_true, n_pred = len(s_true.intersection(s_pred)), len(s_true), len(s_pred)
        prec = n/n_pred
        rec = n/n_true
        f1 = 2*prec*rec/(prec + rec) if prec + rec else 0
        return {
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "n_true": n_true,
            "n_pred": n_pred,
            "n": n
        }

    """"""
    """"""


    def optimize(
            candidate_df: pd.DataFrame,
            prob_df: pd.DataFrame,
            num_kfolds: int,
            weights_filepath_dict: dict,
    ):
        feature_names = get_feature_names()
        X = candidate_df[feature_names].values
        y_preda_list = []
        for mode in weights_filepath_dict.keys():
            fold_y_preda_list = []
            for kfold_index in range(num_kfolds):
                clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
                if mode == 'lgbm':
                    y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
                elif mode == 'lgbm_rank':
                    y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
                else:
                    y_preda = clf.predict_proba(X)[:, 1]
                fold_y_preda_list.append(y_preda)
            mean_preda = np.mean(fold_y_preda_list, axis=0)
            if mode == 'lgbm_rank':  # scaling
                mean_preda = 1 / (1 + np.exp(-mean_preda))
            y_preda_list.append(mean_preda)
        y_preda = np.mean(y_preda_list, axis=0)
        candidate_df["y_preda"] = y_preda

        def f(th):
            _df = candidate_df[y_preda > th]
            if len(_df) == 0:
                return 0
            _gdf = _df.groupby(
                ["audio_id", "seconds"],
                as_index=False
            )["label"].apply(
                lambda _: " ".join(_)
            ).rename(columns={
                "label": "predictions"
            })
            submission_df = pd.merge(
                prob_df[["row_id", "audio_id", "seconds", "birds"]],
                _gdf,
                how="left",
                on=["audio_id", "seconds"]
            )
            submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
            return submission_df.apply(
                lambda row: get_metrics(row["birds"], row["predictions"])["f1"],
                axis=1
            ).mean()

        lb, ub = 0, 1
        for k in range(30):
            th1 = (lb * 2 + ub) / 3
            th2 = (lb + ub * 2) / 3
            if f(th1) < f(th2):
                lb = th1
            else:
                ub = th2
        th = (lb + ub) / 2
        print("-" * 30)
        print("📌best threshold: %f" % th)
        print("best F1: %f" % f(th))

        # nocall injection
        _df = candidate_df[y_preda > th]
        if len(_df) == 0:
            return 0
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(
            lambda _: " ".join(_)
        ).rename(columns={
            "label": "predictions"
        })
        submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"

        _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["y_preda"].sum()
        submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )

        def f_nocall(nocall_th):
            submission_df_with_nocall = submission_df.copy()
            submission_df_with_nocall.loc[(submission_df_with_nocall["y_preda"] < nocall_th)
                                          & (submission_df_with_nocall[
                                                 "predictions"] != "nocall"), "predictions"] += " nocall"
            return submission_df_with_nocall.apply(
                lambda row: get_metrics(row["birds"], row["predictions"])["f1"],
                axis=1
            ).mean()

        lb, ub = 0, 1
        for k in range(30):
            th1 = (lb * 2 + ub) / 3
            th2 = (lb + ub * 2) / 3
            if f_nocall(th1) < f_nocall(th2):
                lb = th1
            else:
                ub = th2
        nocall_th = (lb + ub) / 2
        print("-" * 30)
        print("## nocall injection")
        print("📌best nocall threshold: %f" % nocall_th)
        print("best F1: %f" % f_nocall(nocall_th))

        return th, nocall_th

    """"""
    """"""

    def calc_baseline(prob_df:pd.DataFrame):
        """Calculate the optimal value of F1 score simply based on the threshold alone (without 3rd stage)"""
        columns = BIRD_LIST
        X = prob_df[columns].values
        def f(th):
            n = X.shape[0]
            pred_labels = [[] for i in range(n)]
            I, J = np.where(X > th)
            for i, j in zip(I, J):
                pred_labels[i].append(IDX2BIRD[j])
            for i in range(n):
                if len(pred_labels[i]) == 0:
                    pred_labels[i] = "nocall"
                else:
                    pred_labels[i] = " ".join(pred_labels[i])
            prob_df["pred_labels"] = pred_labels
            return prob_df.apply(
                lambda _: get_metrics(_["birds"], _["pred_labels"])["f1"],
                axis=1
            ).mean()

        lb, ub = 0, 1
        for k in range(30):
            th1 = (2*lb + ub) / 3
            th2 = (lb + 2*ub) / 3
            if f(th1) < f(th2):
                lb = th1
            else:
                ub = th2
        th = (lb + ub) / 2
        print("best th: %.4f" % th)
        print("best F1: %.4f" % f(th))
        return th

    """"""
    """"""


    def make_submission(
            candidate_df: pd.DataFrame,
            prob_df: pd.DataFrame,
            num_kfolds: int,
            th: float,
            nocall_th: float,
            weights_filepath_dict: dict,
            max_distance: int
    ):
        feature_names = get_feature_names()
        X = candidate_df[feature_names].values
        y_preda_list = []
        for mode in weights_filepath_dict.keys():
            fold_y_preda_list = []
            for kfold_index in range(num_kfolds):
                clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
                if mode == 'lgbm':
                    y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
                elif mode == 'lgbm_rank':
                    y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
                else:
                    y_preda = clf.predict_proba(X)[:, 1]
                fold_y_preda_list.append(y_preda)
            mean_preda = np.mean(fold_y_preda_list, axis=0)
            if mode == 'lgbm_rank':  # scaling
                mean_preda = 1 / (1 + np.exp(-mean_preda))
            y_preda_list.append(mean_preda)
        y_preda = np.mean(y_preda_list, axis=0)
        candidate_df["y_preda"] = y_preda

        _df = candidate_df[y_preda > th]
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(
            lambda _: " ".join(_)
        ).rename(columns={
            "label": "predictions"
        })
        submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
        if TARGET_PATH:
            score_df = pd.DataFrame(
                submission_df.apply(
                    lambda row: get_metrics(row["birds"], row["predictions"]),
                    axis=1
                ).tolist()
            )
            print("-" * 30)
            print("BEFORE nocall injection")
            print("CV score on a trained model with train_short_audio (to check the model behavior)")
            print("F1: %.4f" % score_df["f1"].mean())
            print("Recall: %.4f" % score_df["rec"].mean())
            print("Precision: %.4f" % score_df["prec"].mean())

        # nocall injection
        _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["y_preda"].sum()
        submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )
        submission_df.loc[(submission_df["y_preda"] < nocall_th)
                          & (submission_df["predictions"] != "nocall"), "predictions"] += " nocall"
        if TARGET_PATH:
            score_df = pd.DataFrame(
                submission_df.apply(
                    lambda row: get_metrics(row["birds"], row["predictions"]),
                    axis=1
                ).tolist()
            )
            print("-" * 30)
            print("AFTER nocall injection")
            print("CV score on a trained model with train_short_audio (to check the model behavior)")
            print("F1: %.4f" % score_df["f1"].mean())
            print("Recall: %.4f" % score_df["rec"].mean())
            print("Precision: %.4f" % score_df["prec"].mean())

        return submission_df[["row_id", "predictions"]].rename(columns={
            "predictions": "birds"
        })


    """"""
    from soundfile import SoundFile
    def get_audio_info(filepath):
        """Get some properties from  an audio file"""
        with SoundFile(filepath) as f:
            sr = f.samplerate
            frames = f.frames
            duration = float(frames)/sr
        return {"frames": frames, "sr": sr, "duration": duration}
    """"""

    ####################################################
    # Submit Table
    ####################################################

    TARGET_PATH = None
    SAMPLE_SUB_PATH = "D:\\birdclefFirstPlace/input/birdclef-2021/sample_submission.csv"

    r = get_audio_info(TEST_AUDIO_ROOT)
    dur = r["duration"]
    # info = TEST_AUDIO_ROOT.glob("*.ogg").stem.split("_")

    if dur % 5 == 0:
        dur_num = dur/5
    else:
        dur_num = dur/5 + 1

    dur_num = int(dur_num)

    print(dur_num)

    # idlist = [(info[0]+"_"+info[1]+"_" + str((x+1)*5)) for x in range(dur_num)]
    idlist = [(rid + "_" + rlocation + "_" + str((x + 1) * 5)) for x in range(dur_num)]

    print(idlist)

    blist = ['nocall' for x in range(dur_num)]

    print(blist)

    sub_dic = {'row_id': idlist, 'birds': blist}

    sub_df = pd.DataFrame(sub_dic)
    print(sub_df)

    sub_df.to_csv(SAMPLE_SUB_PATH, index=False)

    prob_df = get_prob_df(config, TEST_AUDIO_ROOT, rid, rlocation, rdate)
    print("H" * 100)
    print(prob_df)
    prob_df.info()
    print("H" * 100)
    # candidate extraction
    candidate_df = make_candidates(
        prob_df,
        num_spieces=config.num_spieces,
        num_candidates=config.num_candidates,
        max_distance=config.max_distance,
        num_prob=config.num_prob,
        nocall_threshold=config.nocall_threshold
    )
    # add features
    candidate_df = add_features(
        candidate_df,
        prob_df,
        max_distance=config.max_distance
    )


    submission_df = make_submission(
        candidate_df,
        prob_df,
        num_kfolds=config.num_kfolds,
        th=0.100428,
        nocall_th=0.169353,
        weights_filepath_dict=config.weights_filepath_dict,
        max_distance=config.max_distance
    )

    """"""
    """"""

    submission_df.to_csv("D:\\birdclefFirstPlace/6_step_three/submission.csv", index=False)

    return 0


# classification("D:\\birdclefFirstPlace/input/classifier_test/test1.ogg", '1', 'COR', '20191004', 1)

"""
找到文件夹里最新的，取路径
调用识别
把 csv 换个格式保存
关闭

"""

audiopath = "D:\\tomcatUpload"
filelist = os.listdir(audiopath)
print(filelist)
filelist.sort(key=lambda x: os.path.getmtime((audiopath+"\\"+x)))
print(filelist)
newest = os.path.join(audiopath, filelist[-1])
print(newest)

with open('D:\\apache-tomcat-8.5.66\\webapps\\ROOT\\record.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)
thelocation, thedate, thenum = content.split("@")
thenum = int(thenum)



import shutil
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                # 创建路径
        shutil.move(srcfile, dstfile)          # 移动文件


src_path = newest
dst_path = "D:\\birdclefFirstPlace/input/classifier_test/" + newest.split("\\")[-1]
print(src_path)
print(dst_path)

# if(src_path.split(".")[-1]=="ogg"):
mymovefile(src_path, dst_path)
# elif(src_path.split(".")[-1]=="wav"):
#     print("is wav")
#     tdata, tsamplerate = sf.read(src_path)
#     src_path = src_path.split(".")[0] + ".ogg"
#     print(src_path)
#     dst_path = dst_path.split(".")[0] + ".ogg"
#     sf.write(src_path, tdata, tsamplerate)
#     mymovefile(src_path, dst_path)
#     #exit(0)





# print(newest)
classification(dst_path, '1', thelocation, thedate, thenum)

submitdf = pd.read_csv("D:\\birdclefFirstPlace/6_step_three/submission.csv")

print(list(submitdf["row_id"]))
fromtime = []
totime = []

from time import strftime
from time import gmtime

for i in list(submitdf["row_id"]):
    a, b, c = i.split("_")
    c = int(c)
    fromtime.append(strftime("%H:%M:%S", gmtime(c-5)))
    totime.append(strftime("%H:%M:%S", gmtime(c)))

b1 = []
b2 = []
b3 = []
for i in list(submitdf["birds"]):
    blist = i.split(" ")
    print(blist)
    if len(blist) >= 3:
        b1.append(blist[0])
        b2.append(blist[1])
        b3.append(blist[2])
    elif len(blist) == 2:
        b1.append(blist[0])
        b2.append(blist[1])
        b3.append("nocall")
    else:
        b1.append(blist[0])
        b2.append("nocall")
        b3.append("nocall")

outdata = {'from': fromtime, 'to': totime, 'b1': b1, 'b2': b2, 'b3': b3}
df = pd.DataFrame(outdata)
print(df)

# df.to_csv('D:\\apache-tomcat-8.5.66\\webapps\\ROOT\\result.csv', index=0, header=0)
#
# exit(0)

"""
CREATE TABLE IF NOT EXISTS `ctable`(
   `f` VARCHAR(100) NOT NULL,
   `t` VARCHAR(100) NOT NULL,
   `b1` VARCHAR(100) NOT NULL,
   `b2` VARCHAR(100) NOT NULL,
   `b3` VARCHAR(100) NOT NULL,
   PRIMARY KEY ( `f` )
);
"""

# 导入pymysql包
import pymysql

conn = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="bird")
# 获取一个游标对象
cursor = conn.cursor()
sql = 'DELETE FROM ctable;'
print(sql)
# 执行数据库插入操作
cursor.execute(sql)
# 提交
conn.commit()
# 关闭连接
conn.close()
cursor.close()

# 创建数据库连接
for i, row in df.iterrows():
    print(row)

    conn = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="bird")
    # 获取一个游标对象
    cursor = conn.cursor()
    sql = 'INSERT INTO ctable (f, t, b1, b2, b3) values ("' + row['from'] + '", "' + row['to'] + '", "' + row['b1'] + '", "' + row['b2'] + '", "' + row['b3'] + '");'
    print(sql)
    # 执行数据库插入操作
    cursor.execute(sql)
    # 提交
    conn.commit()
    # 关闭连接
    conn.close()
    cursor.close()


mymovefile(dst_path, "D:\\apache-tomcat-8.5.66\webapps\ROOT\\playaudio.ogg")

# time.sleep(10)

# from pydub import AudioSegment
# from pydub.utils import make_chunks
#
# AudioSegment.converter = r"D:\\birdclefFirstPlace/ffmpeg.exe"
# AudioSegment.ffprobe   = r"D:\\birdclefFirstPlace/ffprobe.exe"
#
# audio = AudioSegment.from_file("D:\\apache-tomcat-8.5.66\webapps\ROOT\\playaudio.ogg", "ogg")
#
# size = 5000  #切割的毫秒数 5s=5000
#
# chunks = make_chunks(audio, size)  #将文件切割为5s一块
#
# for i, chunk in enumerate(chunks):
#     chunk_name = "D:\\apache-tomcat-8.5.66\webapps\ROOT\\playaudios\\playaudio-{0}.ogg".format(i)
#     print(chunk_name)
#     chunk.export(chunk_name, format="ogg")

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

del_files("D:\\apache-tomcat-8.5.66\webapps\ROOT\playaudios\\")

y1, sr1 = lb.load("D:\\apache-tomcat-8.5.66\webapps\ROOT\\playaudio.ogg")
dur1 = lb.get_duration(y1, sr=sr1)
print(dur1)
splitnum = np.ceil(dur1/5).astype(int)

for i in range(splitnum):
    start = i * 5
    duration = 5
    stop = start + duration
    if i == splitnum - 1:
        audio_dst = y1[start * sr1:]
    else:
        audio_dst = y1[start * sr1:stop * sr1]
    sf.write("D:\\apache-tomcat-8.5.66\webapps\ROOT\playaudios\playaudio-" + str(i) + ".ogg", audio_dst, sr1)


conn = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="bird")
# 获取一个游标对象
cursor = conn.cursor()
sql = "UPDATE flag SET isfinished = 'yes' WHERE id = 'idtf';"
print(sql)
# 执行数据库插入操作
cursor.execute(sql)
# 提交
conn.commit()
# 关闭连接
conn.close()
cursor.close()

"""
D:

cd D:/birdclefFirstPlace/venv/Scripts

python D:/birdclefFirstPlace/6_step_three/birdcall_classifier_for_tomcat.py
"""

"""
D: && cd D:/birdclefFirstPlace/venv/Scripts && python D:/birdclefFirstPlace/6_step_three/birdcall_classifier_for_tomcat.py

"""





