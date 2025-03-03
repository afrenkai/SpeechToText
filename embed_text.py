from datasets import Dataset
import matplotlib
matplotlib.use('TkAgg')
from datasets import load_dataset
from consts import PAD, EOS, UNK, TEXT_EMBEDDING_DIM
from embedding_utils import *
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def add_text_to_vocab(ds: Dataset):
    text_sym = [PAD, EOS]
    for row in ds:
        for char in row['text']:
            if char not in text_sym:
                # print(char)
                text_sym.append(char)
    text_sym.append(UNK)
    return text_sym


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, lr=1e-2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.embedding(x)

    def train_embeddings(self, text_seqs, epochs: int = 100, batch_size: int = 32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        losses = []
        dataset = []
        for seq in text_seqs:
            if len(seq) < 2:
                print('little seq, ignoring')
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
                        print('little seq in batch, ignoring')
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


    def viz_embeddings(self, text_vocab):
        embeddings = self.get_embeddings()
        pca = PCA(n_components=2)
        embeddings_2D = pca.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        for i, text in enumerate(text_vocab):
            plt.scatter(embeddings_2D[i, 0], embeddings_2D[i, 1])
            plt.text(embeddings_2D[i, 0], embeddings_2D[i, 1], text, fontsize=12)

        plt.title("Text Embeddings Visualized with PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def save_model(self, path: str='text_embed.pth'):
        torch.save(self.state_dict(), path)

    def load_model(self, path:str ='phoneme_embed.pth'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print('no model found, training from begin lol')

    @staticmethod
    def plot_loss(self, losses: list):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Movement of Training Loss')
        plt.show()

if __name__ == "__main__":
    dataset = load_dataset('Data/')

    dataset = dataset['train']

    text_vocab = add_text_to_vocab(dataset)
    print('vocab made')
    text_dict = symbol_to_idx(text_vocab)
    print('dict made')
    text_seqs = [text_to_seq_per_char(row['text'], text_dict) for row in dataset]
    print(f'seqs made. sample: {text_seqs[0]}')
    vocab_size = len(text_vocab)

    model = TextEmbedding(vocab_size, TEXT_EMBEDDING_DIM)
    model.train_embeddings(text_seqs, epochs=100)
    model.viz_embeddings(text_vocab)
    model.save_model()


