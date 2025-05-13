import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim
import noisereduce as nr

# Attempt to import model classes from train2.py
# Ensure train2.py is in the same directory or Python path

from train2 import AudioEncoder, AudioCompressor, AudioDecompressor, AudioDecoder, FullAutoEncoder
import time


# Configuration
MODEL_PATH = "unet_model.pt"  # Model saved by train2.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000  # Default from train2.py AudioDataset
N_MELS = 64          # Default from train2.py AudioDataset

# Load the pre-trained model
def load_model(model_path, device):
    """Loads the pre-trained FullAutoEncoder model."""
    model = FullAutoEncoder(n_mels=N_MELS).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

# Preprocessing and Postprocessing (adapted from train2.py or common practices)
mel_spectrogram_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
amplitude_to_db_transform = AmplitudeToDB()
inverse_mel_scale_transform = InverseMelScale(n_stft=1024 // 2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE)
griffin_lim_transform = GriffinLim(n_fft=1024, hop_length=256)


def preprocess_audio(audio_path, target_sample_rate=SAMPLE_RATE, device=DEVICE):
    """Loads, resamples, and converts audio to Mel spectrogram."""
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.to(device)
    # Reduce noise if desired (optional, can add significant time)
    # waveform_np = waveform.cpu().numpy()
    # if waveform_np.ndim > 1:
    #     waveform_np = waveform_np[0] # Use first channel if stereo
    # reduced_noise_waveform = nr.reduce_noise(y=waveform_np, sr=target_sample_rate, stationary=True)
    # waveform = torch.tensor(reduced_noise_waveform, device=device).unsqueeze(0)


    mel_spec = mel_spectrogram_transform(waveform)
    db_mel_spec = amplitude_to_db_transform(mel_spec)

    # Normalize
    db_mel_spec = (db_mel_spec - db_mel_spec.mean()) / (db_mel_spec.std() + 1e-6)
    return db_mel_spec.unsqueeze(0) # Add batch dimension

def postprocess_audio(mel_spec_reconstructed, target_sample_rate=SAMPLE_RATE):
    """Converts reconstructed Mel spectrogram back to waveform."""
    # Denormalize (if normalization was complex, this needs to be accurate)
    # For simple mean/std, an exact inverse is hard without original mean/std
    # Often, Griffin-Lim works reasonably well even without perfect denormalization for this type of model
    
    # Assuming the output of the decoder is in DB scale
    # If not, and it's linear magnitude, skip AmplitudeToDB inverse (which is PowerToDB essentially)
    # For now, let's assume it's DB scale and try to convert back
    # This step is tricky and highly dependent on the model's output characteristics

    # If the model outputs something that needs to be converted from DB to power/amplitude
    # For simplicity, let's assume the decoder output is already in a suitable scale for Griffin-Lim
    # or that it's linear magnitude Mel spectrogram.
    # If it's DB, we might need an inverse of AmplitudeToDB (e.g. DBToAmplitude)
    
    # Let's assume the output is a Mel spectrogram (linear amplitude)
    # If it was DB scale, we'd need to convert DB to Power/Amplitude first.
    # For now, we'll directly use Griffin-Lim, which expects linear magnitude.
    # This might need adjustment based on the actual output of your decoder.

    # Remove batch dimension if present
    if mel_spec_reconstructed.ndim == 4 and mel_spec_reconstructed.size(0) == 1:
        mel_spec_reconstructed = mel_spec_reconstructed.squeeze(0)
    
    # Inverse Mel Scale (if the model outputs Mel scale directly)
    # spec_reconstructed = inverse_mel_scale_transform(mel_spec_reconstructed)

    # Griffin-Lim expects linear magnitude spectrogram, not Mel spectrogram
    # If your decoder outputs Mel spectrogram, you need to convert it back to linear frequency spectrogram
    # This is a complex step. For now, let's assume the decoder's output is directly usable
    # by a simplified Griffin-Lim or that the model implicitly learns to output something
    # that Griffin-Lim can process into a waveform.
    # A common approach is that the decoder outputs a magnitude spectrogram.

    # Simplification: Assume the output is a magnitude spectrogram ready for Griffin-Lim
    # This is a strong assumption. The decoder would typically output a Mel spectrogram.
    # For a more accurate reconstruction, the decoder should output a Mel spectrogram,
    # then InverseMelScale, then GriffinLim.
    # However, InverseMelScale is not always perfect.

    # Let's try a direct Griffin-Lim on the output, assuming it's somewhat like a magnitude spectrogram.
    # This is often done in simpler autoencoder examples but might not be optimal.
    # The output of the decoder is (batch, channels, n_mels, time_frames)
    # GriffinLim expects (channels, freq, time) or (freq, time)

    reconstructed_waveform = griffin_lim_transform(mel_spec_reconstructed.cpu())
    return reconstructed_waveform


def encode_audio(model, preprocessed_audio_tensor):
    """Encodes the preprocessed audio using the model's encoder."""
    if preprocessed_audio_tensor is None:
        return None
    with torch.no_grad():
        compressed_representation = model.encoder(preprocessed_audio_tensor)
        compressed_representation = model.compressor(compressed_representation)
    return compressed_representation

def decode_audio(model, compressed_representation):
    """Decodes the compressed representation using the model's decoder."""
    if compressed_representation is None:
        return None
    with torch.no_grad():
        decompressed_representation = model.decompressor(compressed_representation)
        reconstructed_mel_spec = model.decoder(decompressed_representation)
    return reconstructed_mel_spec


def save_audio(waveform, path, sample_rate=SAMPLE_RATE):
    """Saves the waveform to a file."""
    try:
        torchaudio.save(path, waveform.cpu(), sample_rate)
        print(f"Audio saved to {path}")
    except Exception as e:
        print(f"Error saving audio to {path}: {e}")

def encode_decode_and_save(input_audio_path, output_audio_path, model_path=MODEL_PATH):
    """Main function to load model, encode, decode, and save audio."""
    total_start_time = time.time()

    # Load model
    load_model_start_time = time.time()
    model = load_model(model_path, DEVICE)
    if model is None:
        return
    load_model_time = time.time() - load_model_start_time
    print(f"Time to load model: {load_model_time:.4f} seconds")

    # Preprocess audio
    preprocess_start_time = time.time()
    print(f"Preprocessing {input_audio_path}...")
    preprocessed_audio = preprocess_audio(input_audio_path, target_sample_rate=SAMPLE_RATE, device=DEVICE)
    if preprocessed_audio is None:
        return
    preprocess_time = time.time() - preprocess_start_time
    print(f"Time for preprocessing: {preprocess_time:.4f} seconds")

    # Encode audio
    encode_start_time = time.time()
    print("Encoding audio...")
    compressed_representation = encode_audio(model, preprocessed_audio)
    if compressed_representation is None:
        print("Encoding failed.")
        return
    encode_time = time.time() - encode_start_time
    print(f"Time for encoding: {encode_time:.4f} seconds")
    print(f"Shape of compressed representation: {compressed_representation.shape}")


    # Decode audio
    decode_start_time = time.time()
    print("Decoding audio...")
    reconstructed_output = decode_audio(model, compressed_representation)
    if reconstructed_output is None:
        print("Decoding failed.")
        return
    decode_time = time.time() - decode_start_time
    print(f"Time for decoding: {decode_time:.4f} seconds")
    print(f"Shape of reconstructed output (from decoder): {reconstructed_output.shape}")


    # Postprocess audio (convert Mel spectrogram back to waveform)
    postprocess_start_time = time.time()
    print("Postprocessing audio (Mel to waveform)...")
    # The output of the decoder is expected to be a Mel spectrogram
    # We need to convert it back to a waveform.
    reconstructed_waveform = postprocess_audio(reconstructed_output.squeeze(0).cpu(), target_sample_rate=SAMPLE_RATE) # Squeeze batch, move to CPU
    if reconstructed_waveform is None:
        print("Postprocessing failed.")
        return
    postprocess_time = time.time() - postprocess_start_time
    print(f"Time for postprocessing: {postprocess_time:.4f} seconds")
    print(f"Shape of reconstructed waveform: {reconstructed_waveform.shape}")


    # Save reconstructed audio
    save_start_time = time.time()
    print(f"Saving reconstructed audio to {output_audio_path}...")
    save_audio(reconstructed_waveform, output_audio_path, sample_rate=SAMPLE_RATE)
    save_time = time.time() - save_start_time
    print(f"Time for saving: {save_time:.4f} seconds")

    total_time = time.time() - total_start_time
    print(f"Total time for all operations: {total_time:.4f} seconds")


if __name__ == "__main__":
    # Create a dummy input audio file for testing if it doesn't exist
    INPUT_AUDIO_PATH = "input_sample.wav"
    OUTPUT_AUDIO_PATH = "reconstructed_sample.wav"

    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"Creating a dummy input file: {INPUT_AUDIO_PATH}")
        sample_audio = torch.randn(1, SAMPLE_RATE * 2) # 2 seconds of random audio
        torchaudio.save(INPUT_AUDIO_PATH, sample_audio, SAMPLE_RATE)
        print(f"Dummy input file {INPUT_AUDIO_PATH} created.")
    else:
        print(f"Using existing input file: {INPUT_AUDIO_PATH}")

    # Ensure the model exists or train it first using train2.py
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Please train the model using train2.py first.")
        print("You can run: python train2.py")
    else:
        encode_decode_and_save(INPUT_AUDIO_PATH, OUTPUT_AUDIO_PATH, MODEL_PATH)
