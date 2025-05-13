import torch
import torchaudio
import time
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn as nn # Required for model class definitions in train2

# Attempt to import model classes from train2.py
# Ensure train2.py is in the same directory or Python path

from train2 import AudioEncoder, AudioCompressor, AudioDecompressor, AudioDecoder, FullAutoEncoder


# Configuration
MODEL_PATH = "unet_model.pt"  # Model saved by train2.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000  # Default from train2.py AudioDataset
N_MELS = 64          # Default from train2.py AudioDataset

def load_and_preprocess_audio(file_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    """Loads an audio file, converts to mono, resamples, and computes Mel spectrogram."""
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        print("Please ensure a valid audio file (e.g., 'example.mp3') exists at the specified path.")
        return None

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Mel spectrogram transforms
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, hop_length=512, win_length=1024) # Typical hop/win lengths
    db_transform = AmplitudeToDB()

    mel = mel_transform(waveform)
    mel_db = db_transform(mel)

    # Reshape to match model input: (batch, channels, n_mels, time_frames)
    # train2.py training loop uses .unsqueeze(1) on dataset output.
    return mel_db.unsqueeze(0).to(DEVICE)


def main_inference(audio_file_path="example.mp3"):
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")

    # Initialize model components (as done in train2.py)
    try:
        model_encoder = AudioEncoder().to(DEVICE)
        model_compressor = AudioCompressor().to(DEVICE)
        model_decompressor = AudioDecompressor().to(DEVICE)
        model_decoder = AudioDecoder().to(DEVICE)

        model = FullAutoEncoder(
            model_encoder,
            model_compressor,
            model_decompressor,
            model_decoder
        ).to(DEVICE)
    except Exception as e:
        print(f"Error initializing model components: {e}")
        print("This might be due to issues with importing from train2.py or the placeholder classes.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Model file not found: {MODEL_PATH}")
        print("Please ensure the model is trained and saved correctly by train2.py.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    model.eval()

    # Load and preprocess audio
    print(f"Loading and preprocessing audio: {audio_file_path}")
    input_tensor = load_and_preprocess_audio(audio_file_path)
    if input_tensor is None:
        return

    print(f"Input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        # Encoding part
        # Corresponds to: encoded = self.encoder(x); compressed = self.compressor(encoded)
        start_time_encode = time.perf_counter()
        encoded_output = model.encoder(input_tensor)
        compressed_representation = model.compressor(encoded_output)
        end_time_encode = time.perf_counter()
        encode_time = end_time_encode - start_time_encode
        print(f"Encoding time: {encode_time:.6f} seconds")
        print(f"Shape after encoder: {encoded_output.shape}")
        print(f"Shape after compressor (compressed representation): {compressed_representation.shape}")


        # Decoding part
        # Corresponds to: decompressed = self.decompressor(compressed); reconstructed = self.decoder(decompressed)
        start_time_decode = time.perf_counter()
        decompressed_output = model.decompressor(compressed_representation)
        reconstructed_mel = model.decoder(decompressed_output)
        end_time_decode = time.perf_counter()
        decode_time = end_time_decode - start_time_decode
        print(f"Decoding time: {decode_time:.6f} seconds")
        print(f"Shape after decompressor: {decompressed_output.shape}")
        print(f"Shape after decoder (reconstructed Mel): {reconstructed_mel.shape}")

if __name__ == "__main__":
    # Assuming 'example.mp3' exists in the same directory as the script,
    # or provide a full path to an audio file.
    # The workspace structure indicates 'ai/example.mp3' exists.
    main_inference(audio_file_path="/home/inf151841/nanochi/ai2/ai/example.mp3")
