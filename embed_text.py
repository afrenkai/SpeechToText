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
    def __init__(self, vocab_size, embed_dim, lr=1e-3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.embedding(x)

    def train_embeddings(self, text_tensor: torch.IntTensor, epochs: int = 100):
        target_vec = torch.randn_like(self.embedding(text_tensor))

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            outputs = self(text_tensor)
            loss = self.criterion(outputs, target_vec)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch {epoch}, L: {loss.item()}')
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





if __name__ == "__main__":
    dataset = load_dataset('Data/')

    dataset = dataset['train']

    text_vocab = add_text_to_vocab(dataset)
    text_dict = symbol_to_idx(text_vocab)
    trenches = (text_to_seq_per_char('apple', text_dict))
    vocab_size = len(text_vocab)
    model = TextEmbedding(vocab_size, TEXT_EMBEDDING_DIM)


    model.train_embeddings(trenches, epochs=100)

    model.viz_embeddings(text_vocab)



