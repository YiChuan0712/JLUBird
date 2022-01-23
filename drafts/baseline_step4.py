import os
import pandas as pd
import torch
import librosa
import numpy as np

# Global vars
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # seconds
SPEC_SHAPE = (224, 224)  # height x width
FMIN = 20
FMAX = 16000
# Load metadata file
train = pd.read_csv('D:/birdclef-2021/train_metadata.csv', )
# Second, assume that birds with the most training samples are also the most common
# A species needs at least 200 recordings with a rating above 4 to be considered common
birds_count = {}
for bird_species, count in zip(train.primary_label.unique(),
                               train.groupby('primary_label')['primary_label'].count().values):
    birds_count[bird_species] = count
most_represented_birds = [key for key, value in birds_count.items()]

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())

# Let's see how many species and samples we have left
print('NUMBER OF SPECIES IN TRAIN DATA:', len(LABELS))
print('NUMBER OF SAMPLES IN TRAIN DATA:', len(TRAIN))
print('LABELS:', most_represented_birds)


# First, get a list of soundscape files to process.
# We'll use the test_soundscape directory if it contains "ogg" files
# (which it only does when submitting the notebook),
# otherwise we'll use the train_soundscape folder to make predictions.
def list_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.rsplit('.', 1)[-1] in ['ogg']]


test_audio = list_files('D:/birdclef-2021/test_soundscapes')
if len(test_audio) == 0:
    test_audio = list_files('D:/birdclef-2021/train_soundscapes')
print('{} FILES IN TEST SET.'.format(len(test_audio)))
path = test_audio[0]
data = path.split(os.sep)[-1].rsplit('.', 1)[0].split('_')
print('FILEPATH:', path)
print('ID: {}, SITE: {}, DATE: {}'.format(data[0], data[1], data[2]))
# This is where we will store our results
pred = {'row_id': [], 'birds': []}
model = torch.load("./model/efficientnet-b3.pth")
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
# Analyze each soundscape recording
# Store results so that we can analyze them later
data = {'row_id': [], 'birds': []}
for path in test_audio:
    path = path.replace("\\", "/")
    # Open file with Librosa
    # Split file into 5-second chunks
    # Extract spectrogram for each chunk
    # Predict on spectrogram
    # Get row_id and birds and store result
    # (maybe using a post-filter based on location)
    # The above steps are just placeholders, we will use mock predictions.
    # Our "model" will predict "nocall" for each spectrogram.
    sig, rate = librosa.load(path, sr=SAMPLE_RATE)
    # Split signal into 5-second chunks
    # Just like we did before (well, this could actually be a seperate function)
    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)
    # Get the spectrograms and run inference on each of them
    # This should be the exact same process as we used to
    # generate training samples!
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
        # Normalize to match the value range we used during training.
        # That's something you should always double check!
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