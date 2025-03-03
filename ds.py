from typing import Tuple
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from embedding_utils import text_to_seq_per_char, symbol_to_idx
from embed_text import add_text_to_vocab
from speech_utils import SpeechConverter
from consts import PAD

class SpeechDataset(Dataset):
    def __init__(self, dataset: Dataset, symb_idx: dict, num_mels=128,
                 text_col='normalized_text', phoneme_col='phonemes', text_to_seq_fn=text_to_seq_per_char, ):
        self.dataset = dataset
        self.num_mels = num_mels
        self.text_col = text_col
        self.phoneme_col = phoneme_col
        self.text_to_seq_fn = text_to_seq_fn
        self.speech_converter = SpeechConverter(self.num_mels)
        self.symb_idx = symb_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.IntTensor, torch.Tensor]:
        text = self.dataset[idx][self.text_col]
        audio_waveform = self.dataset[idx]['audio']['array']
        # sr is constant for the dataset, use speech_utils.sr
        # sampling_rate = self.hf_dataset[idx]['audio']['sampling_rate']

        # Apply text_to_seq_fn to the text
        text_seq = self.text_to_seq_fn(text, self.symb_idx)

        # Processing the wave-form
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        #     mel_transform = MelSpectrogram(sampling_rate, n_mels=self.num_mels)
        # mel_spec = mel_transform(audio_waveform)
        mel_spec = self.speech_converter.convert_to_mel_spec(audio_waveform)

        return text_seq, mel_spec


def speech_collate_fn(batch, symb_idx):
    # sort the batch based on input text (this is needed for pack_padded_sequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    text_seqs, mel_specs = zip(*batch)
    text_seq_lens = [text_seq.shape[-1] for text_seq in text_seqs]  # batch first

    mel_specs_t = []
    mel_spec_lens = []
    max_mel_seq = -1
    for mel_spec in mel_specs:
        mel_specs_t.append(mel_spec.T)
        true_mel_size = mel_spec.shape[-1]
        mel_spec_lens.append(true_mel_size)
        if true_mel_size > max_mel_seq:
            max_mel_seq = true_mel_size
    # need to know max size to pad to/ generate stop token input
    stop_token_targets = []
    for i in range(len(mel_specs)):
        stop_token_target = torch.zeros(max_mel_seq)
        true_mel_size = mel_spec_lens[i]
        stop_token_target[true_mel_size - 1:] = 1
        stop_token_targets.append(stop_token_target)

    # pad sequence so pytorch can batch them together
    # alternatives using the minimum from the batch
    # this is using the right padding for samples that have seq_len < max_batch_seq_len
    padded_text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=symb_idx.get(PAD))
    padded_mel_specs = pad_sequence(mel_specs_t, batch_first=True, padding_value=0)
    text_seq_lens = torch.IntTensor(text_seq_lens)
    mel_spec_lens = torch.IntTensor(mel_spec_lens)
    stop_token_targets = torch.stack(stop_token_targets)
    print("In collate", padded_mel_specs.shape, stop_token_targets.shape)
    return padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets


def get_data_loader(dataset: Dataset, batch_size, shuffle=True, num_workers=0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=speech_collate_fn,
                      num_workers=num_workers)


def load_data(batch_size, mel_bins=128, subsample_ratio=None):
    # load dataset
    dataset = load_dataset('Data/')['train']
    if subsample_ratio is not None:  # Used for testing model arch
        hf_dataset = dataset.train_test_split(train_size=subsample_ratio)['train']
    dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)
    # split dataset into training and (validation+test) set
    hf_split_datadict = dataset.train_test_split(test_size=0.2)
    hf_train_dataset = hf_split_datadict['train']
    # split (validation+test) dataset into validation and test set
    hf_dataset = hf_split_datadict['test']
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.5)
    hf_val_dataset = hf_split_datadict['train']
    hf_test_dataset = hf_split_datadict['test']
    print(f'Dataset Sizes: Train ({len(hf_train_dataset)}), Val ({len(hf_val_dataset)}), Test ({len(hf_test_dataset)})')
    # convert hf_dataset to pytorch datasets
    train_text_vocab = add_text_to_vocab(hf_train_dataset)
    val_text_vocab = add_text_to_vocab(hf_val_dataset)
    test_text_vocab = add_text_to_vocab(hf_test_dataset)
    train_text_symb_idx = symbol_to_idx(train_text_vocab)
    val_text_symb_idx = symbol_to_idx(val_text_vocab)
    test_text_symb_idx = symbol_to_idx(test_text_vocab)
    train_ds = SpeechDataset(hf_train_dataset, num_mels=mel_bins, symb_idx = train_text_symb_idx)
    val_ds = SpeechDataset(hf_val_dataset, num_mels=mel_bins, symb_idx = val_text_symb_idx)
    test_ds = SpeechDataset(hf_test_dataset, num_mels=mel_bins, symb_idx = test_text_symb_idx)
    # convert datasets to dataloader
    train_dl = get_data_loader(train_ds, batch_size, num_workers=3)
    val_dl = get_data_loader(val_ds, batch_size, shuffle=False, num_workers=1)
    test_dl = get_data_loader(test_ds, batch_size, shuffle=False, num_workers=1)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    train_dl, val_dl, test_dl = load_data(64, mel_bins=80, subsample_ratio=None )
    print(val_dl.dataset[3])