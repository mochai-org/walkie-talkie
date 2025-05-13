import torch
import torchaudio
import time
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim
from torchaudio.functional import DB_To_Amplitude
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
    # Ensure n_fft >= win_length. Set n_fft = win_length if not otherwise specified.
    # The error indicated n_fft=400 and win_length=1024.
    # Default n_fft for MelSpectrogram is 400.
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, win_length=1024)
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

        # Convert reconstructed Mel spectrogram back to waveform and save
        print("Reconstructing waveform from Mel spectrogram...")
        
        # reconstructed_mel shape is (batch, channels, n_mels, time_frames)
        # Assuming batch_size=1 and channels=1 as input is mono and unsqueezed to (1,1,H,W)
        # Squeeze batch and channel dimensions: (n_mels, time_frames)
        reconstructed_mel_squeezed = reconstructed_mel.squeeze(0).squeeze(0).cpu() 

        # 1. Convert from dB to Power Spectrogram
        # MelSpectrogram produces power spec, AmplitudeToDB converts power to dB.
        # DBToAmplitude(ref=1.0, power=1.0) inverts X_db = 10 * log10(X_power)
        db_to_power_transform = torchaudio.transforms.DB_To_Amplitude(ref=1.0, power=1.0)
        reconstructed_power_mel_spec = db_to_power_transform(reconstructed_mel_squeezed)

        # 2. Convert from Mel scale to linear frequency scale
        # n_fft = 1024, so n_stft (number of linear frequency bins) = 1024 // 2 + 1 = 513
        inverse_mel_scale_transform = torchaudio.transforms.InverseMelScale(
            n_stft=1024 // 2 + 1, 
            n_mels=N_MELS, 
            sample_rate=SAMPLE_RATE,
            norm="slaney", # Common normalization, check if matches MelSpectrogram if issues
            mel_scale="slaney" # Common mel scale, check if matches MelSpectrogram
        )
        reconstructed_linear_power_spec = inverse_mel_scale_transform(reconstructed_power_mel_spec)

        # 3. Use Griffin-Lim to convert the linear power spectrogram to waveform
        griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=1024, 
            hop_length=512, 
            win_length=1024, 
            power=2.0 # Input is a power spectrogram
        )
        reconstructed_waveform = griffin_lim_transform(reconstructed_linear_power_spec)
        
        # Ensure waveform is 2D (channels, time) for saving. GriffinLim output is (time_samples).
        # Add channel dimension for mono audio.
        if reconstructed_waveform.ndim == 1:
            reconstructed_waveform = reconstructed_waveform.unsqueeze(0)

        # 4. Save the waveform
        output_filename_mp3 = "reconstructed_audio.mp3"
        output_filename_wav = "reconstructed_audio.wav"
        try:
            torchaudio.save(output_filename_mp3, reconstructed_waveform, SAMPLE_RATE)
            print(f"Reconstructed audio saved to {output_filename_mp3}")
        except Exception as e:
            print(f"Error saving audio as MP3: {e}")
            print(f"This might be due to torchaudio backend issues or ffmpeg not being available for MP3 encoding.")
            print(f"Attempting to save as .wav instead: {output_filename_wav}")
            try:
                torchaudio.save(output_filename_wav, reconstructed_waveform, SAMPLE_RATE)
                print(f"Reconstructed audio saved to {output_filename_wav}")
            except Exception as e_wav:
                print(f"Error saving audio as .wav: {e_wav}")

if __name__ == "__main__":
    # Assuming 'example.mp3' exists in the same directory as the script,
    # or provide a full path to an audio file.
    # The workspace structure indicates 'ai/example.mp3' exists.
    main_inference(audio_file_path="example.mp3")
