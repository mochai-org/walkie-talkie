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
import torch.nn.functional as F

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
        # Augmentacja: zmiana głośności (Vol)
        self.vol_augment = Vol(gain=6, gain_type="db")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(audio_path)

        # Konwersja do mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample w razie potrzeby
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Augmentacja tylko w trybie treningowym
        if self.train:
            if random.random() < self.transform_prob:
                gain_db = random.uniform(-6, 6)
                vol_transform = Vol(gain=gain_db, gain_type="db")
                waveform = vol_transform(waveform)

        # Tworzenie Mel-spektrogramu
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        # Ewentualnie łączenie z MFCC
        if self.use_mfcc:
            mfcc = self.mfcc_transform(waveform).squeeze(0)
            stacked = torch.cat((mel_db.squeeze(0), mfcc), dim=0)
        else:
            stacked = mel_db.squeeze(0)

        return stacked


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Można dorzucić czwarty poziom w razie potrzeby
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.final_conv(x)
        return x


class AudioCompressor(nn.Module):
    def __init__(self):
        super(AudioCompressor, self).__init__()
        # Załóżmy, że końcowy wymiar z AudioEncoder to (128, 8, 8). Dostosuj do rzeczywistych rozmiarów.
        self.flat_dim = 128 * 8 * 8
        self.fc_reduce1 = nn.Linear(self.flat_dim, 64)
        self.fc_reduce2 = nn.Linear(64, 2)

    def forward(self, x):
        # x zakładamy jako [batch, 128, 8, 8]
        x = x.view(x.size(0), -1)  # spłaszczenie
        x = self.fc_reduce1(x)
        z = self.fc_reduce2(x)     # wynik 2-float
        return z


class AudioDecompressor(nn.Module):
    def __init__(self):
        super(AudioDecompressor, self).__init__()
        self.fc_expand1 = nn.Linear(2, 64)
        self.fc_expand2 = nn.Linear(64, 128 * 8 * 8)

    def forward(self, z):
        x = self.fc_expand1(z)
        x = self.fc_expand2(x)
        x = x.view(x.size(0), 128, 8, 8)
        return x


class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.final_conv(x)
        return x


class FullAutoEncoder(nn.Module):
    def __init__(self, encoder, compressor, decompressor, decoder):
        super(FullAutoEncoder, self).__init__()
        self.encoder = encoder
        self.compressor = compressor
        self.decompressor = decompressor
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        compressed = self.compressor(encoded)
        decompressed = self.decompressor(compressed)
        reconstructed = self.decoder(decompressed)
        return reconstructed, compressed, encoded, decompressed


# ------------------------------------------------------------------------
# Główna funkcja treningowa (z pętlą i mixed precision)
# ------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/inf151841/nanochi/ai/input/cv/pl/clips"

    # Parametry wg poprzedniego kodu
    batch_size = 256
    num_epochs = 300
    learning_rate = 3e-3
    train_val_split = 0.9

    # Załadowanie datasetu
    full_dataset = AudioDataset(data_dir=data_dir, train=True)
    total_len = len(full_dataset)
    train_len = int(train_val_split * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    # Dataloadery
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=52
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=52
    )

    # Używamy U-Net
    model_encoder = AudioEncoder().to(device)
    model_compressor = AudioCompressor().to(device)
    model_decompressor = AudioDecompressor().to(device)
    model_decoder = AudioDecoder().to(device)
    model = FullAutoEncoder(
        model_encoder,
        model_compressor,
        model_decompressor,
        model_decoder
    ).to(device)

    # Kryterium
    mse_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    # Optimizer i scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Mixed precision
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float("inf")

    # Główna pętla treningowa
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for data in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            data = data.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                reconstructed, compressed, encoded, decompressed = model(data)
                
                # Loss rekonstrukcji
                mse_rec = mse_crit(reconstructed, data)
                l1_rec = l1_crit(reconstructed, data)
                rec_loss = 0.7 * mse_rec + 0.3 * l1_rec

                # Loss kompresji (porównanie zdekompresowanego z wyjściem encodera)
                mse_comp = mse_crit(decompressed, encoded)
                l1_comp = l1_crit(decompressed, encoded)
                comp_loss = 0.7 * mse_comp + 0.3 * l1_comp

                # Łączny błąd
                loss = rec_loss + comp_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                data = data.to(device).unsqueeze(1)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    output, compressed = model(data)
                    mse_loss = mse_crit(output, data)
                    l1_loss = l1_crit(output, data)
                    loss = 0.7 * mse_loss + 0.3 * l1_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Scheduler
        scheduler.step(val_loss)

        # Zapisujemy model każdorazowo, a jeśli jest najlepszy, można to odnotować
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/home/inf151841/nanochi/ai2/ai/unet_model.pt")
            print("Zapisano nowy najlepszy model (val_loss).")

    print("Trening zakończony. Model zapisany jako unet_model.pt")


if __name__ == "__main__":
    main()
