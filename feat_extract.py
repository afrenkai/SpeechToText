import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from consts import PAD, UNK, EOS

# def extract_features(batch):
#     features = []
#     for example in batch["audio"]:
#         y, sr = example["array"], example["sampling_rate"]
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         mfccs = np.mean(mfccs, axis=1)  # Take mean across time
#         pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#         pitch = np.nanmean(pitch) if np.any(~np.isnan(pitch)) else 0  # idk how to do nans, this should be good i think
#         energy = np.mean(librosa.feature.rms(y=y))  # Root mean square energy
#         duration = len(y) / sr  # Duration in seconds
#
#         # phoneme_embedding =  WIP
#         # thinking something like using get_phoneme_embedding from the trained model per phoneme
#
#         features.append(np.concatenate([mfccs, [pitch, energy, duration]]))
#
#     return {"features": features}


# if __name__ == "__main__":
#     dataset = load_dataset('Data/')
#     dataset = dataset['train']
#     dataset = dataset.map(extract_features, batched=True, batch_size=32, num_proc=4)
#     print('done mapping')
#     X = np.vstack(dataset["features"])
#     y = np.array(dataset["phonemes"])  # might swap to another label val not sure how to tackle yet
#     print(X.shape)
#     print(y.shape)