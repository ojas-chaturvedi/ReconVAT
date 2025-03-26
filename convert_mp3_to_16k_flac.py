#!/opt/homebrew/bin/python3
"""
Name: convert_mp3_to_16k_flac.py
Purpose: Convert MP3 audio files to 16kHz FLAC audio files.
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

from pydub import AudioSegment
import soundfile as sf


def convert_mp3_to_16k_flac(input_mp3_path, output_flac_path):
    # Load mp3 using pydub
    audio = AudioSegment.from_mp3(input_mp3_path)

    # Set frame rate (downsample to 16k)
    audio = audio.set_frame_rate(16000).set_channels(1)  # mono channel

    # Save as FLAC using soundfile
    sf.write(output_flac_path, audio.get_array_of_samples(), 16000, format="FLAC")

    print(f"Converted: {input_mp3_path} → {output_flac_path}")


if __name__ == "__main__":
    # Example usage
    input_path = "Ode_to_Joy.mp3"
    output_path = "Application/Input/Ode_to_Joy.flac"
    convert_mp3_to_16k_flac(input_path, output_path)
