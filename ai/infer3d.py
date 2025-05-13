import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim
import noisereduce as nr
from pydub import AudioSegment
from train3d import ConvAutoencoderWithMultiStageVQ  # Twój model treningowy

def infer(input_file="example.mp3", output_wav="reconstructed.wav"):
    # Ustawienia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 16000
    n_mels = 64
    n_fft = 1024
    hop_length = 256

    # Inicjalizacja modelu i ładowanie checkpointu
    model = ConvAutoencoderWithMultiStageVQ(
        n_stages=2,
        num_embeddings=128,
        embedding_dim=4,   # UWAGA: musi być zgodne z checkpointem
        commitment_cost=0.25
    ).to(device)
    checkpoint = torch.load("/home/inf151841/nanochi/ai2/ai/autoencoder_3d_final.pt", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Przygotowanie transformacji audio -> mel-spektrogram
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    db_transform = AmplitudeToDB()

    # Wczytanie i przygotowanie pliku audio
    waveform, sr = torchaudio.load(input_file)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel = mel_transform(waveform)         # (1, n_mels, time)
    mel_db = db_transform(mel).squeeze(0)   # (n_mels, time)

    # Ponieważ podczas treningu model otrzymywał 3-klatkowe zestawy (poprzedni, bieżący, następny),
    # tutaj powielamy mel-spektrogram 3 razy, by stworzyć tensor o kształcie (1, 1, 3, n_mels, time)
    mel_context = mel_db.unsqueeze(0).repeat(3, 1, 1)  # (3, n_mels, time)
    mel_context = mel_context.unsqueeze(0).unsqueeze(1)  # (1, 1, 3, n_mels, time)

    with torch.no_grad():
        output_tensor, _ = model(mel_context)
        output_tensor = output_tensor.squeeze(0).squeeze(0)  # (n_mels, time)

    # Zamiana z dB na amplitudę
    output_ampl = torch.pow(10.0, (output_tensor / 20.0))

    # Odwrócenie transformacji mel-spektrogram -> widmo
    inv_melscale = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    mag_spec = inv_melscale(output_ampl.unsqueeze(0))
    griffinlim = GriffinLim(n_fft=n_fft, hop_length=hop_length)
    reconstructed_wave = griffinlim(mag_spec)

    torchaudio.save(output_wav, reconstructed_wave, sample_rate)
    print(f"Inference zakończone. Zapisano WAV: {output_wav}")
    return output_wav

def remove_noise(input_wav, output_wav, sample_rate=16000):
    # Wczytanie audio
    waveform, sr = torchaudio.load(input_wav)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).numpy()
    # Używamy pierwszych 0.5 sekundy jako profil szumu
    noise_profile = audio_np[:sample_rate // 2]
    reduced_audio = nr.reduce_noise(y=audio_np, sr=sample_rate, y_noise=noise_profile, stationary=True)
    reduced_waveform = torch.tensor(reduced_audio).unsqueeze(0)
    torchaudio.save(output_wav, reduced_waveform, sample_rate)
    print(f"Redukcja szumu zakończona. Zapisano WAV: {output_wav}")
    return output_wav

def convert_to_mp3(input_wav, output_mp3, bitrate="32k"):
    # Używamy pydub do konwersji WAV do MP3
    audio = AudioSegment.from_wav(input_wav)
    audio.export(output_mp3, format="mp3", bitrate=bitrate)
    print(f"Konwersja: {input_wav} -> {output_mp3} (bitrate: {bitrate})")
    return output_mp3

if __name__ == "__main__":
    wav_file = infer("example.mp3", "reconstructed.mp3")
    denoised_wav = remove_noise(wav_file, "reconstructed_denoised.mp3")
