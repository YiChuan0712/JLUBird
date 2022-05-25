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
import pymysql

"""
志放请注意：所有path最后都加“/”，包括upload_path
"""

# working_path = f"D:/birdclefFirstPlace/6_step_three/"
# model_path = f"D:/birdclefFirstPlace/input/models/"
# upload_path = "D:/tomcatUpload/"
# web_path = "D:/apache-tomcat-8.5.66/webapps/ROOT/"
# meta_path = "D:/birdclefFirstPlace/input/birdclef-2021/"

working_path = f"D:/bird/"
model_path = f"D:/bird/models/"
upload_path = "D:/tomcatUpload/"
web_path = "D:/tomcat/webapps/ROOT/"
meta_path = "D:/bird/birdclef-2021/"

mysql_host = "localhost"
mysql_port = 3306
mysql_user = "root"
mysql_passwd = "root"
mysql_db = "bird"


def classification(TEST_AUDIO_ROOT, rid, rlocation, rdate, mnum):

    BIRD_LIST = ['acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro', 'amegfi', 'amekes', 'amepip', 'amered', 'amerob', 'amewig', 'amtspa', 'andsol1', 'annhum', 'astfly', 'azaspi1', 'babwar', 'baleag', 'balori', 'banana', 'banswa', 'banwre1', 'barant1', 'barswa', 'batpig1', 'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher', 'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo', 'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1', 'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho', 'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow', 'bobfly1', 'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla', 'brncre', 'brnjay', 'brnthr', 'brratt1', 'brwhaw', 'brwpar1', 'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan', 'buhvir', 'bulori', 'burwar1', 'bushti', 'butsal1', 'buwtea', 'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar', 'carchi', 'carwre', 'casfin', 'caskin', 'caster1', 'casvir', 'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1', 'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob', 'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 'colcha1', 'coltro1', 'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1', 'comrav', 'comyel', 'coohaw', 'cotfly1', 'cowscj1', 'cregua1', 'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly', 'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro', 'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa', 'gadwal', 'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo', 'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1', 'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 'grcfly', 'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel', 'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra', 'grycat', 'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr', 'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa', 'houwre', 'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo', 'larspa', 'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1', 'lesvio1', 'linspa', 'linwoo1', 'littin1', 'lobdow', 'lobgna5', 'logshr', 'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre', 'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1', 'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar', 'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit', 'obnthr1', 'ocbfly1', 'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar', 'osprey', 'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2', 'phivir', 'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1', 'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut', 'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1', 'rehbar1', 'relpar', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinkin1', 'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum', 'rtlhum', 'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar', 'rucwar1', 'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1', 'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 'scamac1', 'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2', 'sinwre1', 'sltred', 'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan', 'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1', 'squcuc1', 'stbori', 'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2', 'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin', 'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc', 'tropar', 'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa', 'warvir', 'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea', 'westan', 'wewpew', 'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa', 'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1', 'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc', 'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1', 'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla', 'yehcar1', 'yelgro', 'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir']
    # print(BIRD_LIST)
    BIRD2IDX = {bird: idx for idx, bird in enumerate(BIRD_LIST)}
    BIRD2IDX['nocall'] = -1
    IDX2BIRD = {idx: bird for bird, idx in BIRD2IDX.items()}

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
                'lgbm': [(working_path + f"lgbm_{kfold_index}.pkl") for kfold_index in range(self.num_kfolds)],
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
                Path(model_path + "birdclef_resnest50_fold0_epoch_27.pth"),  # id36
                Path(model_path + "birdclef_resnest50_fold0_epoch_13.pth"),  # id51
                Path(model_path + "birdclef_resnest50_fold0_epoch_33.pth"),
                # id58
                Path(model_path + "birdclef_resnest50_fold1_epoch_34.pth"),
                # id59
                Path(model_path + "birdclef_resnest50_fold2_epoch_34.pth"),
                # id60
                Path(model_path + "birdclef_resnest50_fold3_epoch_20.pth"),
                # id61
                Path(model_path + "birdclef_resnest50_fold4_epoch_34.pth"),
                # id62
                Path(model_path + "birdclef_resnest50_fold0_epoch_78.pth"),
                # id97
                Path(model_path + "birdclef_resnest50_fold0_epoch_84.pth"),
                # id97
                Path(model_path + "birdclef_resnest50_fold1_epoch_27.pth"),
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

        # candidate_df = candidate_df.dropna(subset=["row_id"])
        #
        # print(candidate_df)
        # candidate_df.info()

        candidate_df["target"] = candidate_df.apply(
            lambda row: IDX2BIRD[row["bird_id"]] in set(row["birds"].split()),
            axis=1
        )
        candidate_df["label"] = candidate_df["bird_id"].map(IDX2BIRD)
        return candidate_df

    """"""
    """"""

    def load_metadata():
        meta_df = pd.read_csv(meta_path + "train_metadata.csv")
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



    class BirdCLEFDataset(Dataset):
        def __init__(self, data, sr=32000, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast",
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
            self.npy_save_root = Path(working_path + "data")

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
        thresh = thresh  # or THRESH
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
            #if (not os.path.exists(prob_filepath)) or (TARGET_PATH is None):  # Always calculate when no cash is available or when submitting.
            nets = [load_net(checkpoint_path.as_posix())]
            pred_probas = predict(nets, test_data, names=False)

            df = sub_df  # pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id", "birds"])


            df["audio_id"] = df["row_id"].apply(lambda _: int(_.split("_")[0]))
            df["site"] = df["row_id"].apply(lambda _: _.split("_")[1])
            df["seconds"] = df["row_id"].apply(lambda _: int(_.split("_")[2]))
            assert len(data) == len(pred_probas)
            n = len(data)
            audio_id_to_date = {}
            audio_id_to_site = {}

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


    """"""
    """"""


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

    r = get_audio_info(TEST_AUDIO_ROOT)
    dur = r["duration"]

    if dur % 5 == 0:
        dur_num = dur/5
    else:
        dur_num = dur/5 + 1

    dur_num = int(dur_num)

    print(dur_num)

    idlist = [(rid + "_" + rlocation + "_" + str((x + 1) * 5)) for x in range(dur_num)]

    print(idlist)

    blist = ['nocall' for x in range(dur_num)]

    print(blist)

    sub_dic = {'row_id': idlist, 'birds': blist}

    sub_df = pd.DataFrame(sub_dic)
    print(sub_df)

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

    return submission_df

"""
找到文件夹里最新的，取路径
调用识别
把 csv 换个格式保存
关闭

"""

audiopath = upload_path
filelist = os.listdir(audiopath)
print(filelist)
# filelist.sort(key=lambda x: os.path.getmtime((audiopath+"/"+x)))
filelist.sort(key=lambda x: os.path.getmtime((audiopath+x)))
print(filelist)
newest = os.path.join(audiopath, filelist[-1])
print(newest)

with open(web_path+"record.txt", 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)
thelocation, thedate, thenum = content.split("@")
thenum = int(thenum)

# conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
# # 获取一个游标对象
# cursor = conn.cursor()
# sql = 'select * from uinput;'
# print(sql)
# # 执行数据库插入操作
# cursor.execute(sql)
# userinput = cursor.fetchall()
#
# thelocation, thedate, thenum = userinput[0][0], userinput[0][1], userinput[0][2]
# print(len(thelocation))
# print(thedate)
# print(thenum)
# thenum = int(thenum)
# # 关闭连接
# conn.close()
# cursor.close()





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
dst_path = web_path + "temp/" + newest.split("/")[-1]  #"D:\\birdclefFirstPlace/input/classifier_test/" + newest.split("\\")[-1]
print(src_path)
print(dst_path)


mymovefile(src_path, dst_path)


# print(newest)
submitdf = classification(dst_path, '1', thelocation, thedate, thenum)

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

# conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
# # 获取一个游标对象
# cursor = conn.cursor()
# sql = 'DELETE FROM ctable;'
# print(sql)
# # 执行数据库插入操作
# cursor.execute(sql)
# # 提交
# conn.commit()
# # 关闭连接
# conn.close()
# cursor.close()

# 创建数据库连接
ctable = ""
for i, row in df.iterrows():
    print(row)
    ctable += row['from'] + '@' + row['to'] + '@' + row['b1'] + '@' + row['b2'] + '@' + row['b3'] + "\n"
    # conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
    # # 获取一个游标对象
    # cursor = conn.cursor()
    # sql = 'INSERT INTO ctable (f, t, b1, b2, b3) values ("' + row['from'] + '", "' + row['to'] + '", "' + row['b1'] + '", "' + row['b2'] + '", "' + row['b3'] + '");'
    # print(sql)
    # # 执行数据库插入操作
    # cursor.execute(sql)
    # # 提交
    # conn.commit()
    # # 关闭连接
    # conn.close()
    # cursor.close()
with open(web_path + "ctable.txt", "w") as f:
    f.write(ctable)

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


mymovefile(dst_path, web_path + "playaudio.ogg")

del_files(web_path + "playaudios/")

y1, sr1 = lb.load(web_path + "playaudio.ogg")
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
    sf.write(web_path + "playaudios\playaudio-" + str(i) + ".ogg", audio_dst, sr1)






# conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
# # 获取一个游标对象
# cursor = conn.cursor()
# sql = 'DELETE FROM bcount;'
# print(sql)
# # 执行数据库插入操作
# cursor.execute(sql)
# # 提交
# conn.commit()
# # 关闭连接
# conn.close()
# cursor.close()

bdlst = submitdf['birds'].values.tolist()
# print(len(bdlst))
print(bdlst)

newbdlst = []
for i in bdlst:
    i = i.split(" ")
    for ii in i:
        if ii != "nocall":
            newbdlst.append(ii)
print(newbdlst)

wordcount = {}
for word in newbdlst:
    wordcount[word] = wordcount.get(word, 0)+1
wordcount = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)[:10]
print(wordcount)

bcount = ""
for i in wordcount:
    print(i)
    bcount += i[0] + "@" + str(i[1]) + "\n"
    # conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
    # # 获取一个游标对象
    # cursor = conn.cursor()
    # sql = 'INSERT INTO bcount (name, count) values ("' + i[0] + '", ' + str(i[1]) + ');'
    # print(sql)
    # # 执行数据库插入操作
    # cursor.execute(sql)
    # # 提交
    # conn.commit()
    # # 关闭连接
    # conn.close()
    # cursor.close()
# bcount = bcount.sort(key=lambda s: (-s[1]))
# print(bcount)
with open(web_path + "bcount.txt", "w") as f:
    f.write(bcount)

# conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, passwd=mysql_passwd, db=mysql_db)
# # 获取一个游标对象
# cursor = conn.cursor()
# sql = "UPDATE flag SET isfinished = 'yes' WHERE id = 'idtf';"
# print(sql)
# # 执行数据库插入操作
# cursor.execute(sql)
# # 提交
# conn.commit()
# # 关闭连接
# conn.close()
# cursor.close()
with open(web_path + "isfinished.txt", "w") as f:
    f.write("yes")







