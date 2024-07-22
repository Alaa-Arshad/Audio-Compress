from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

def compress_audio_with_dct(input_path, output_path, compression_format="wav", bitrate="64k", target_sr=16000):
    # Load the audio file
    audio = AudioSegment.from_file(input_path, format=compression_format)

    # Downsample the audio to the target sampling rate
    audio = audio.set_frame_rate(target_sr)

    # Extract audio samples
    samples = np.array(audio.get_array_of_samples())

    # Apply DCT to the samples
    compressed_samples = dct(samples, type=2, axis=0, norm='ortho')

    # Inverse DCT to reconstruct the compressed audio
    reconstructed_samples = idct(compressed_samples, type=2, axis=0, norm='ortho')

    # Convert back to 16-bit integer format
    reconstructed_samples = np.clip(reconstructed_samples, -2*15, 2*15 - 1).astype(np.int16)

    # Create a new AudioSegment with the reconstructed samples
    compressed_audio = AudioSegment(reconstructed_samples.tobytes(), frame_rate=target_sr, sample_width=2, channels=1)

    # Export the compressed audio
    compressed_audio.export(output_path, format=compression_format, bitrate=bitrate)

    print(f"Audio compression complete. Saved to {output_path}")

# Example usage:
input_audio_path = "C:/Users/Eng.ALAA Arshad/Downloads/audio1.wav"
output_audio_path = "compressed_audio_with_dct.wav"

# Load the original audio
original_audio = AudioSegment.from_file(input_audio_path, format="wav")
fs_original = original_audio.frame_rate

# Compress the audio with DCT
compress_audio_with_dct(input_audio_path, output_audio_path)

# Load the compressed audio
compressed_audio = AudioSegment.from_file(output_audio_path, format="wav")
fs_compressed = compressed_audio.frame_rate

# Plot the waveforms (you can use the plot_waveforms function from your original code)
def plot_waveforms(original_waveform, compressed_waveform, fs_original, fs_compressed):
    # Use the minimum length of the two waveforms
    min_length = min(len(original_waveform), len(compressed_waveform))

    # Plot the original waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(min_length) / fs_original, original_waveform[:min_length])
    plt.title('Original Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot the compressed waveform
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(min_length) / fs_compressed, compressed_waveform[:min_length])
    plt.title('Compressed Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Example usage:
input_audio_path = "C:/Users/Eng.ALAA Arshad/Downloads/audio1.wav"
output_audio_path = "compressed_audio_with_dct.wav"

# Load the original audio
original_audio = AudioSegment.from_file(input_audio_path, format="wav")
fs_original = original_audio.frame_rate

# Compress the audio with DCT
compress_audio_with_dct(input_audio_path, output_audio_path)

# Load the compressed audio
compressed_audio = AudioSegment.from_file(output_audio_path, format="wav")
fs_compressed = compressed_audio.frame_rate

# Plot the waveforms
plot_waveforms(np.array(original_audio.get_array_of_samples()), np.array(compressed_audio.get_array_of_samples()), fs_original, fs_compressed)