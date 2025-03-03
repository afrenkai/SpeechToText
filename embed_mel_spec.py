import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader
from ds import load_data
from consts import AUDIO_EMBEDDING_DIM
import os
class AudioEmbed(nn.Module):
    def __init__(self, input_dim, embed_dim = AUDIO_EMBEDDING_DIM, lr=1e-3):
        super().__init__()
        # the hubert method
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding = 1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding = 1)
        self.pool = nn.AdaptiveAvgPool1d(1) # seq turned into 1 vec
        self.fc = nn.Linear(128, embed_dim) # projection into embed space
        self.optimizer = optim.AdamW(self.parameters(), lr = lr)
        self.criterion = nn.TripletMarginLoss(margin=1.0)


    def forward(self, x):
        x = torch.relu(torch.BatchNorm(self.conv1(x)))
        x = torch.relu(torch.BatchNorm(self.conv2(x)))
        x = self.pool(x).squeeze(-1) #bs x 128
        x = self.fc(x)
        return x

    #TODO: clean this up and improve logging. plot loss won't work as it is right now. not a prio but should do
    def train_embed(self, dl: DataLoader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        losses = []
        for epoch in tqdm(range(epochs)):
            running_loss = []
            self.train()

            for batch in dl:
                audio_data = batch[2].to(device)
                self.optimizer.zero_grad()
                embed = self.forward(audio_data)
                anchor, pos, neg = embed[::3], embed[1::3], embed[2::3]
                loss = self.criterion(anchor, pos, neg)

                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())
            print(f'epoch {epoch}, loss = {running_loss[-1]}')
            # self.plot_loss(running_loss)

    def get_embed(self, dl: DataLoader):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in dl:
                audio_data = batch[2]
                emb = self.forward(audio_data)
                embeddings.append(emb.cpu.numpy) # convert to np array so i can vstack
        return np.vstack(embeddings)

    def viz_embeddings(self, dl: DataLoader):
        embeddings = self.get_embed(dl)
        pca = PCA(n_components=2)
        embeddings_reduced = pca.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], alpha=0.6)
        plt.title("Audio Embeddings Visualized with PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def save_model(self, path: str = 'audio_embed.pth'):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str = 'audio_embed.pth'):
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
    train_dl, _, _ = load_data(64, mel_bins=80, subsample_ratio=None)
    sample_audio = train_dl.dataset[0][2]  # Example audio tensor
    input_dim = sample_audio.shape[0]  # Number of mel bins
    assert(sample_audio.shape == 80) # should be mel_bins
    model = AudioEmbed(input_dim)
    model.train_embed(train_dl, epochs=50)
    model.visualize_embeddings(train_dl)