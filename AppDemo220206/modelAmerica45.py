import os
import pandas as pd
import torch
import librosa
import numpy as np

from efficientnet_pytorch import EfficientNet

def testClassification(filepath):

    # Global vars
    RANDOM_SEED = 1337
    SAMPLE_RATE = 32000
    SIGNAL_LENGTH = 5  # seconds
    SPEC_SHAPE = (224, 224)  # height x width
    FMIN = 20
    FMAX = 16000

    LABELS = ['amecro', 'amegfi', 'amerob', 'balori', 'belkin1', 'bkcchi', 'blujay', 'bobfly1', 'bucmot2', 'cangoo', 'chswar', 'clcrob', 'comgra', 'comyel', 'crfpar', 'eastow', 'eawpew', 'gockin', 'grekis', 'grhcha1', 'grycat', 'haiwoo', 'hofwoo1', 'norcar', 'norfli', 'norwat', 'obnthr1', 'orcpar', 'orfpar', 'ovenbi1', 'plawre1', 'rebwoo', 'reevir1', 'rewbla', 'rtlhum', 'rubwre1', 'rucwar', 'runwre1', 'sonspa', 'sthwoo1', 'swaspa', 'whcpar', 'woothr', 'yebsap', 'yehcar1']

# This is where we will store our results
    pred = {'row_id': [], 'birds': []}
    model = torch.load("D:/JLUbird/models/efficientnet-b3-45.pth")
    model.eval()

    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = {'row_id': [], 'birds': []}
    path = filepath
    path = path.replace("\\", "/")

    sig, rate = librosa.load(path, sr=SAMPLE_RATE)

    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)

    seconds, scnt = 0, 0
    for chunk in sig_splits:
        # Keep track of the end time of each chunk
        seconds += 5
        # Get the spectrogram
        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                  sr=SAMPLE_RATE,
                                                  n_fft=2048,
                                                  hop_length=hop_length,
                                                  n_mels=SPEC_SHAPE[0],
                                                  fmin=FMIN,
                                                  fmax=FMAX)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im = transform(im)
        print(im.shape)
        im.unsqueeze_(0)
        # 没有这句话会报错
        im = im.to(device)
        # Predict
        p = model(im)[0]
        print(p.shape)
        # Get highest scoring species
        idx = p.argmax()
        print(idx)
        species = LABELS[idx]
        print(species)
        score = p[idx]
        print(score)
        # Prepare submission entry
        spath = path.split('/')[-1].rsplit('_', 1)[0]
        print(spath)
        data['row_id'].append(path.split('/')[-1].rsplit('_', 1)[0] +
                              '_' + str(seconds))
        # Decide if it's a "nocall" or a species by applying a threshold
        if score > 0.75:
            data['birds'].append(species)
            scnt += 1
        else:
            data['birds'].append('nocall')
    print('SOUNSCAPE ANALYSIS DONE. FOUND {} BIRDS.'.format(scnt))
    # Make a new data frame and look at a few "results"
    results = pd.DataFrame(data, columns=['row_id', 'birds'])
    results.head()
    # Convert our results to csv
    results.to_csv("submission.csv", index=False)
    return results

#testClassification("3.wav")