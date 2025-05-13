# train.py

import os
import random
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------------------------------
# 1. Dataset i collate_fn – jak w poprzednich kodach
# ------------------------------------------------

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
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".mp3") or f.endswith(".wav")
        ]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.train = train
        self.transform_prob = transform_prob
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        if self.use_mfcc:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, n_mfcc=n_mfcc
            )
        self.vol_augment = torchaudio.transforms.Vol(gain=6, gain_type="db")

    def __len__(self):
        return len(self.files)

    def _load_and_mel(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Augmentacja w trybie treningowym
        import random
        if self.train and random.random() < self.transform_prob:
            gain_db = random.uniform(-6, 6)
            vol_transform = torchaudio.transforms.Vol(gain=gain_db, gain_type="db")
            waveform = vol_transform(waveform)

        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        if self.use_mfcc:
            mfcc = self.mfcc_transform(waveform).squeeze(0)
            stacked = torch.cat((mel_db.squeeze(0), mfcc), dim=0)
        else:
            stacked = mel_db.squeeze(0)

        return stacked

    def __getitem__(self, idx):
        prev_idx = max(0, idx - 1)
        next_idx = min(len(self.files) - 1, idx + 1)

        mel_prev = self._load_and_mel(prev_idx)
        mel_curr = self._load_and_mel(idx)
        mel_next = self._load_and_mel(next_idx)

        min_time = min(mel_prev.shape[-1], mel_curr.shape[-1], mel_next.shape[-1])
        mel_prev = mel_prev[..., :min_time]
        mel_curr = mel_curr[..., :min_time]
        mel_next = mel_next[..., :min_time]

        triplet = torch.stack([mel_prev, mel_curr, mel_next], dim=0)  # (3, freq, time)
        return triplet

def custom_collate_fn(batch):
    max_time = max(item.shape[-1] for item in batch)
    padded_batch = []
    for triplet_mel in batch:
        time_dim = triplet_mel.shape[-1]
        if time_dim < max_time:
            pad_size = max_time - time_dim
            triplet_mel = F.pad(triplet_mel, (0, pad_size))
        padded_batch.append(triplet_mel)
    return torch.stack(padded_batch, dim=0)  # (batch, 3, freq, max_time)


# ------------------------------------------------
# 2. Blok 3D i MultiStageVectorQuantizer
# ------------------------------------------------

class Initial3DConv(nn.Module):
    """
    Pierwsza warstwa 3D: (B,1,3,freq,time) -> (B,16,freq/2, time/2)
    """
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3, 3),
            stride=(3, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn3d = nn.BatchNorm3d(16)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, 1, 3, freq, time)
        x = self.conv3d(x)  # -> (B,16,1,freq/2,time/2)
        x = self.bn3d(x)
        x = self.act(x)
        x = x.squeeze(2)    # -> (B,16, freq/2, time/2)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        B, C, H, W = x.shape
        flat_x = x.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        distances = (
            flat_x.pow(2).sum(1, keepdim=True)
            - 2 * flat_x @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=1)
        )
        encoding_indices = distances.argmin(1)
        # ...
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        quantized = self.embeddings(encoding_indices).view(B,H,W,C).permute(0,3,1,2)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        return quantized, loss

class MultiStageVectorQuantizer(nn.Module):
    def __init__(self, n_stages=2, num_embeddings=128, embedding_dim=4, commitment_cost=0.25):
        super().__init__()
        self.n_stages = n_stages
        self.vq_blocks = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(n_stages)
        ])

    def forward(self, x):
        residual = x
        total_loss = 0.0
        quant_sum = torch.zeros_like(x)
        for vq in self.vq_blocks:
            quantized, loss = vq(residual)
            total_loss += loss
            quant_sum = quant_sum + quantized
            residual = x - quant_sum
        return quant_sum, total_loss


# ------------------------------------------------
# 3. Rozdzielamy: Encoder, Decoder, FullAutoencoder
# ------------------------------------------------

class ConvAutoencoderEncoder(nn.Module):
    def __init__(self,
                 n_stages=2,
                 num_embeddings=128,
                 embedding_dim=4,
                 commitment_cost=0.25):
        super().__init__()
        self.initial_3d = Initial3DConv()  # -> (B,16,freq/2,time/2)

        self.encoder_2d = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),  # freq/2->freq/4
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1),    # freq/4->freq/8
            nn.BatchNorm2d(4),
            nn.GELU()
        )
        self.multi_vq = MultiStageVectorQuantizer(
            n_stages=n_stages,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

    def forward(self, x):
        """
        x: (B,1,3,freq,time)
        """
        z_3d = self.initial_3d(x)   # -> (B,16,freq/2,time/2)
        z_2d = self.encoder_2d(z_3d)# -> (B,4,freq/8,time/8)  np.
        z_quant, vq_loss = self.multi_vq(z_2d)
        return z_quant, vq_loss

class ConvAutoencoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z_quant):
        """
        z_quant: (B,4,freq/8, time/8)
        """
        return self.decoder(z_quant)

class FullAutoencoder(nn.Module):
    def __init__(self,
                 n_stages=2,
                 num_embeddings=128,
                 embedding_dim=4,
                 commitment_cost=0.25):
        super().__init__()
        self.encoder = ConvAutoencoderEncoder(n_stages, num_embeddings, embedding_dim, commitment_cost)
        self.decoder = ConvAutoencoderDecoder()

    def forward(self, x):
        z_quant, vq_loss = self.encoder(x)
        out = self.decoder(z_quant)
        return out, vq_loss


# ------------------------------------------------
# 4. Główna pętla treningowa
# ------------------------------------------------

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/inf151841/nanochi/ai/input/cv/pl/clips")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--dataset_percent", type=float, default=0.5)
    parser.add_argument("--out_ckpt", type=str, default="/home/inf151841/nanochi/ai2/ai/full_3d_autoencoder.pt")
    args = parser.parse_args()

    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"    {arg}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    full_dataset = AudioDataset(data_dir=args.data_dir, train=True)
    total_len = len(full_dataset)
    print(f"Dataset: {total_len} files.")
    if total_len == 0:
        print("No files found. Exiting.")
        sys.exit(0)

    max_dataset_size = int(total_len * args.dataset_percent)
    all_indices = list(range(total_len))
    random.shuffle(all_indices)
    used_indices = all_indices[:max_dataset_size]
    reduced_dataset = Subset(full_dataset, used_indices)
    print(f"Used {len(reduced_dataset)} samples (dataset_percent={args.dataset_percent})")

    train_len = int(args.train_val_split * len(reduced_dataset))
    val_len = len(reduced_dataset) - train_len
    train_dataset, val_dataset = random_split(reduced_dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )

    model = FullAutoencoder(n_stages=2, num_embeddings=128, embedding_dim=4, commitment_cost=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    mse_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0

        for data in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            data = data.unsqueeze(1).to(device)  # (batch,1,3,freq,time)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output, vq_loss = model(data)
                if output.shape[-1] > data.shape[-1]:
                    output = output[..., : data.shape[-1]]
                target = data[:, :, 1, ...]  # (batch,1,freq,time)
                mse_loss = mse_crit(output, target)
                l1_loss = l1_crit(output, target)
                loss = 0.7 * mse_loss + 0.3 * l1_loss + vq_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Val
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                data = data.unsqueeze(1).to(device)
                with torch.cuda.amp.autocast():
                    output, vq_loss = model(data)
                    if output.shape[-1] > data.shape[-1]:
                        output = output[..., : data.shape[-1]]
                    target = data[:, :, 1, ...]
                    mse_loss = mse_crit(output, target)
                    l1_loss = l1_crit(output, target)
                    loss = 0.7*mse_loss + 0.3*l1_loss + vq_loss
                val_loss_total += loss.item()
        val_loss = val_loss_total / len(val_loader)

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"Best model saved at epoch {epoch} (val_loss={val_loss:.4f})")

    print("Training done.")

if __name__ == "__main__":
    main()
