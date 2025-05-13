import os
import random
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import (
    MelSpectrogram, AmplitudeToDB, Vol, MFCC
)
from tqdm import tqdm

# ------------------------------------------------------------------------
# Simple local attention block (small context window)
# ------------------------------------------------------------------------
class SmallLocalAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        hidden_dim = in_channels // 2  # reduce dimension
        self.conv_q = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv_k = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv_v = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)  # project back to in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.conv_q(x).view(b, -1, h*w)      # (B, hidden_dim, H*W)
        k = self.conv_k(x).view(b, -1, h*w)      # (B, hidden_dim, H*W)
        v = self.conv_v(x).view(b, -1, h*w)      # (B, hidden_dim, H*W)
        attn_scores = torch.bmm(q.permute(0, 2, 1), k) / (q.shape[1] ** 0.5)  # (B, H*W, H*W)
        attn_map = self.softmax(attn_scores)
        out = torch.bmm(attn_map, v.permute(0, 2, 1))  # (B, H*W, hidden_dim)
        out = out.permute(0, 2, 1).view(b, -1, h, w)
        return x + self.proj_out(out)

# ------------------------------------------------------------------------
# Dataset z podstawową augmentacją i wygenerowanym podziałem train/val
# ------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        sample_rate=16000,
        n_mels=64,
        train=True,
        transform_prob=0.1,
        use_mfcc=False,
        n_mfcc=40
    ):
        """
        data_dir   : folder zawierający pliki MP3
        sample_rate: docelowa częstotliwość próbkowania
        n_mels     : liczba mel-filterbanków
        train      : flaga określająca, czy to zbiór treningowy
        transform_prob: prawdopodobieństwo przeprowadzenia augmentacji
        use_mfcc   : flaga określająca, czy używać MFCC
        n_mfcc     : liczba współczynników MFCC
        """
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".mp3")
        ]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.train = train
        self.transform_prob = transform_prob
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc

        self.mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db_transform = AmplitudeToDB()
        if self.use_mfcc:
            self.mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
        # Augmentacja: zmiana głośności
        self.vol_augment = Vol(gain=6, gain_type="db")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(audio_path)

        # Konwersja do mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Augmentacja tylko w trybie treningowym
        if self.train:
            if random.random() < self.transform_prob:
                gain_db = random.uniform(-6, 6)
                vol_transform = Vol(gain=gain_db, gain_type="db")
                waveform = vol_transform(waveform)

        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        if self.use_mfcc:
            mfcc = self.mfcc_transform(waveform).squeeze(0)
            stacked = torch.cat((mel_db.squeeze(0), mfcc), dim=0)
        else:
            stacked = mel_db.squeeze(0)
        return stacked

# ------------------------------------------------------------------------
# Squeeze-and-Excitation (SE)
# ------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

# ------------------------------------------------------------------------
# Autoencoder with SE blocks and small local attention
# ------------------------------------------------------------------------
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        # x is expected as (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten for distance computation: (B, C, H, W) -> (B*H*W, C)
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Find nearest embeddings
        distances = (
            flat_x.pow(2).sum(1, keepdim=True)
            - 2 * flat_x @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=1)
        )
        encoding_indices = distances.argmin(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Convert indices back to (B, C, H, W)
        quantized = self.embeddings(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2)

        # Compute vector-quantization losses
        # Fix the mismatch by permuting x to match quantized
        # so both are (B, C, H, W)
        x_fixed = x
        if x.shape != quantized.shape:
            # Transpose x if needed so that dimension 2,3 match quantized
            # e.g., if your x is (B, C, W, H) by mistake:
            x_fixed = x.permute(0, 1, 3, 2)

        e_latent_loss = F.mse_loss(quantized.detach(), x_fixed)
        q_latent_loss = F.mse_loss(quantized, x_fixed.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss

class TemporalGRU(nn.Module):
    def __init__(self, in_channels, freq_size):
        super().__init__()
        input_size = in_channels * freq_size  # e.g. 16 * 32 = 512
        hidden_dim = input_size // 2
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True, bidirectional=True)
        self.out_channels = hidden_dim * 2
        self.freq_size = freq_size

    def forward(self, x):
        b, c, h, w = x.shape
        # Make sure H matches freq_size
        if h != self.freq_size:
            raise ValueError(f"Expected freq_size={self.freq_size}, got H={h}")
        # Flatten (C × H) -> input_size for GRU
        x_reshaped = x.view(b, c * h, w).transpose(1, 2)  # shape (B, W, input_size)
        gru_output, _ = self.gru(x_reshaped)             # shape (B, W, 2*hidden_dim)
        out = gru_output.transpose(1, 2)                  # (B, 2*hidden_dim, W)
        if (c * h) != self.out_channels:
            out = nn.Conv1d(self.out_channels, c * h, 1)(out)
        out = out.view(b, c, h, w)
        return out

class ConvAutoencoderWithGRUMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            SEBlock(16),
            TemporalGRU(16, 32),  # replaces SmallLocalAttention
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SEBlock(64),
            TemporalGRU(64, 16),  # another memory block
            nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            SEBlock(4),
        )

        self.vq = VectorQuantizer(num_embeddings=128, embedding_dim=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SEBlock(64),
            TemporalGRU(64, 16),  # again replacing attention with GRU
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            SEBlock(16),
            TemporalGRU(16, 32),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        out = self.decoder(z_q)
        if out.shape[-1] > x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return out, vq_loss

# ------------------------------------------------------------------------
# Funkcja treningowa z walidacją
# ------------------------------------------------------------------------
def train_model(model, train_dl, val_dl, device, epochs, optimizer, criterion, scheduler, alpha=0.5):
    """
    criterion: MSELoss (we will combine with L1Loss)
    alpha    : weighting factor for MSE in combined loss
    """
    l1_criterion = nn.L1Loss()
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_dl, desc=f"Train Epoch {epoch}"):
            data = data.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output, vq_loss = model(data)
            mse_loss = criterion(output, data)
            l1_loss = l1_criterion(output, data)
            loss = alpha * mse_loss + (1 - alpha) * l1_loss + vq_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_dl, desc=f"Val Epoch {epoch}"):
                data = data.to(device).unsqueeze(1)
                output, vq_loss = model(data)
                mse_loss = criterion(output, data)
                l1_loss = l1_criterion(output, data)
                loss = alpha * mse_loss + (1 - alpha) * l1_loss + vq_loss
                val_loss += loss.item()
        val_loss /= len(val_dl)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/home/inf151841/nanochi/ai2/ai/autoencoder_final.pt")
            print("Zapisano nowy najlepszy model (val_loss).")

# ------------------------------------------------------------------------
# Custom collate function
# ------------------------------------------------------------------------
def custom_collate_fn(batch):
    max_time = max(item.shape[-1] for item in batch)
    padded_batch = []
    for mel_db in batch:
        time_dim = mel_db.shape[-1]
        if time_dim < max_time:
            pad_size = max_time - time_dim
            mel_db = torch.nn.functional.pad(mel_db, (0, pad_size))
        padded_batch.append(mel_db)
    return torch.stack(padded_batch, dim=0)

# ------------------------------------------------------------------------
# Główna funkcja main
# ------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/inf151841/nanochi/ai/input/cv/pl/clips"

    # Reduce batch size further to cut memory usage
    batch_size = 256  # lowered from 64
    num_epochs = 300
    learning_rate = 3e-3
    train_val_split = 0.9

    full_dataset = AudioDataset(data_dir=data_dir, train=True)
    total_len = len(full_dataset)
    train_len = int(train_val_split * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=52
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=52
    )

    model = ConvAutoencoderWithGRUMemory().to(device)
    mse_crit = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # (Optional: enable mixed precision training)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, vq_loss = model(data)
                mse_loss = mse_crit(output, data)
                l1_loss = nn.L1Loss()(output, data)
                loss = 0.7 * mse_loss + 0.3 * l1_loss + vq_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device).unsqueeze(1)
                with torch.cuda.amp.autocast():
                    output, vq_loss = model(data)
                    mse_loss = mse_crit(output, data)
                    l1_loss = nn.L1Loss()(output, data)
                    loss = 0.7 * mse_loss + 0.3 * l1_loss + vq_loss
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)

        torch.save(model.state_dict(), "/home/inf151841/nanochi/ai2/ai/autoencoder_final.pt")
    print("Trening zakończony. Model zapisany jako autoencoder_final.pt")

if __name__ == "__main__":
    main()
