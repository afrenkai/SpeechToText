from datasets import load_dataset
import os

def save_ds(path: str = "Data"):
    os.makedirs(path, exist_ok=True)
    ds = load_dataset("bookbot/ljspeech_phonemes")
    ds.save_to_disk(path)
    print(f'saved dataset to {path}')

if __name__ == "__main__":
    save_ds()