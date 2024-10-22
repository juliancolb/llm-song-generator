import pyttsx3
import librosa
import soundfile as sf
import numpy as np
import os

def generate_procedural_song(lyrics):
    lines = lyrics.strip().split('\n')

    # Define pitch shifts in semitones for each line based on I-IV-V-I progression in C major
    # C (0 semitones), F (+5 semitones), G (+7 semitones), C (0 semitones)
    pitch_shifts = [0, 5, 7, 0] * ((len(lines) // 4) + 1)
    pitch_shifts = pitch_shifts[:len(lines)]  # Adjust to the number of lines

    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech

    filenames = []

    # Remove old audio files if any
    for i in range(len(lines)):
        filename = f'line_{i}.wav'
        if os.path.exists(filename):
            os.remove(filename)

    # Generate speech audio for each line
    for i, line in enumerate(lines):
        filename = f'line_{i}.wav'
        filenames.append(filename)
        engine.save_to_file(line, filename)

    # Run the speech engine to save all files
    engine.runAndWait()

    # List to hold pitch-shifted audio arrays
    audio_list = []

    # Process each audio file
    for i, filename in enumerate(filenames):
        # Load audio
        y, sr = librosa.load(filename, sr=None)  # Use original sampling rate

        # Shift pitch according to the chord progression
        n_steps = pitch_shifts[i]

        # Corrected function call with keyword arguments
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        # Append to audio list
        audio_list.append(y_shifted)

    # Concatenate all audio segments
    combined_audio = np.concatenate(audio_list)

    # Save the final audio file
    sf.write('procedural_song.wav', combined_audio, sr)
    print("Procedural song generated and saved as 'procedural_song.wav'.")

# Example usage:
lyrics = """Twinkle twinkle little star
How I wonder what you are
Up above the world so high
Like a diamond in the sky"""

generate_procedural_song(lyrics)
