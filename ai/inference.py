import torch
import torchaudio
import torch.nn as nn
import numpy as np
from torchaudio.transforms import (
    MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim
)
from train2 import UNet  # Nowy model U-Net


def encode_audio(
    input_file="example.mp3",
    encoded_file="encoded_tensor.pt",
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load("/home/inf151841/nanochi/ai2/ai/unet_model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    db_transform = AmplitudeToDB()

    waveform, sr = torchaudio.load(input_file)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel = mel_transform(waveform)
    mel_db = db_transform(mel).to(device)      # [1, n_mels, time]
    mel_db = mel_db.unsqueeze(0)              # [1, 1, n_mels, time]

    with torch.no_grad():
        output_db = model(mel_db)  # [1, 1, n_mels, time]

    torch.save(output_db.cpu(), encoded_file)
    print(f"Saved encoded tensor to {encoded_file}")


def decode_audio(
    encoded_file="encoded_tensor.pt",
    output_file="reconstructed.mp3",
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_db = torch.load(encoded_file, map_location=device)  # [1, 1, n_mels, time]

    output_db_cpu = output_db.squeeze(0).squeeze(0).cpu()      # [n_mels, time]
    output_ampl = torch.pow(10.0, (output_db_cpu / 20.0))

    inv_melscale = InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    mag_spec = inv_melscale(output_ampl.unsqueeze(0))
    griffinlim = GriffinLim(n_fft=n_fft, hop_length=hop_length)
    reconstructed_wave = griffinlim(mag_spec)

    torchaudio.save(output_file, reconstructed_wave, sample_rate)
    print(f"Reconstructed audio saved to {output_file}")

if __name__ == "__main__":
    encode_audio(
        input_file="example.mp3",
        encoded_file="encoded_tensor.pt",
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    decode_audio(
        encoded_file="encoded_tensor.pt",
        output_file="reconstructed.mp3",
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )