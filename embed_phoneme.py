from datasets import Dataset
import matplotlib
matplotlib.use('TkAgg')
from datasets import load_dataset
from consts import PAD, EOS, UNK, PHONEME_EMBEDDING_DIM
from embedding_utils import *
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def add_phonemes_to_vocab(ds: Dataset):
    phoneme_sym = [PAD, EOS]
    for row in ds:
        for char in row['phonemes']:
            if char not in phoneme_sym:
                # print(char)
                phoneme_sym.append(char)
    phoneme_sym.append(UNK)
    return phoneme_sym


class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, lr=1e-3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.embedding(x)

    def train_embeddings(self, phoneme_seqs, epochs: int = 20, batch_size: int = 32):
        #cuda stuff not needed for now but ill def want it later
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # print(f"Vocabulary Size: {vocab_size}")
        # print(f"Max Index in Phonemes Dict: {max(phonemes_dict.values())}")


        losses = []
        dataset = []
        for seq in phoneme_seqs:
            if len(seq) < 2:
                continue
            dataset.append(torch.tensor(seq, dtype=torch.long, device=device))

        for epoch in tqdm(range(epochs)):
            total_loss = 0
            self.train()

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i: i + batch_size]

                self.optimizer.zero_grad()
                loss = 0
                for seq in batch:
                    if len(seq) < 2:
                        continue
                    inputs = seq[:-1]
                    targets = seq[1:]

                    outputs = self.embedding(inputs)
                    loss += self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataset)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        self.plot_loss(losses)


    def get_embeddings(self):
        return self.embedding.weight.detach().numpy()


    def viz_embeddings(self, phonemes_vocab):
        embeddings = self.get_embeddings()
        pca = PCA(n_components=2)
        embeddings_2D = pca.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        for i, phoneme in enumerate(phonemes_vocab):
            plt.scatter(embeddings_2D[i, 0], embeddings_2D[i, 1])
            plt.text(embeddings_2D[i, 0], embeddings_2D[i, 1], phoneme, fontsize=12)

        plt.title("Phoneme Embeddings Visualized with PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def save_model(self, path: str='phoneme_embed.pth'):
        torch.save(self.state_dict(), path)

    def load_model(self, path:str ='phoneme_embed.pth'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print('no model found, training from begin lol')

    def plot_loss(self, losses: list):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Movement of Training Loss')
        plt.show()



if __name__ == "__main__":
    dataset = load_dataset('Data/')

    dataset = dataset['train']

    phonemes_vocab = add_phonemes_to_vocab(dataset)
    print('vocab made')
    # print(phonemes_vocab)
    phonemes_dict = symbol_to_idx(phonemes_vocab)
    print('dict made')

    phoneme_seqs = [text_to_seq_per_char(row['phonemes'], phonemes_dict) for row in dataset]
    
    print(f'seqs made. sample: {phoneme_seqs[0]}')
    # gaming = (seq_to_text(epic, phonemes_vocab))
    # print(gaming)
    
    vocab_size = len(phonemes_vocab)
    model = PhonemeEmbedding(vocab_size, PHONEME_EMBEDDING_DIM)

    model.train_embeddings(phoneme_seqs, epochs=100)

    model.viz_embeddings(phonemes_vocab)
    model.save_model()


