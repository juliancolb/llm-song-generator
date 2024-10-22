import pyttsx3
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile
import shutil

def generate_procedural_song_with_chorus(lyrics, output_prefix='procedural_song'):
    lines = lyrics.strip().split('\n')
    
    # Define chord progression: I, IV, V, I repeating
    chord_progression = ['I', 'IV', 'V', 'I']
    
    # Define triads for each chord in semitone shifts relative to C
    # C major: C (0), E (+4), G (+7)
    # F major: F (+5), A (+9), C (+12)
    # G major: G (+7), B (+11), D (+14)
    triads = {
        'I': [0, 4, 7],
        'IV': [5, 9, 12],
        'V': [7, 11, 14]
    }
    
    # Assign chords to lines
    chords = [chord_progression[i % len(chord_progression)] for i in range(len(lines))]
    
    # Initialize pitch shifts for each singer
    # Singer 1: root, Singer 2: third, Singer 3: fifth
    singer_pitch_shifts = {
        'singer1': [],
        'singer2': [],
        'singer3': []
    }
    
    for chord in chords:
        triad = triads[chord]
        singer_pitch_shifts['singer1'].append(triad[0])
        singer_pitch_shifts['singer2'].append(triad[1])
        singer_pitch_shifts['singer3'].append(triad[2])
    
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    
    # Prepare temporary directory for audio files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Lists to hold audio data for each singer
        audio_data = {
            'singer1': [],
            'singer2': [],
            'singer3': []
        }
        sample_rate = None  # To store the sampling rate
        
        print("Generating speech and applying pitch shifts for each singer...")
        
        # Initialize list to hold mixed audio segments
        mixed_audio_segments = []
        
        for i, line in enumerate(lines):
            # Create a temporary file for the speech
            temp_filename = os.path.join(temp_dir, f'temp_line_{i}.wav')
            
            # Generate speech for the line and save to temp file
            engine.save_to_file(line, temp_filename)
            engine.runAndWait()
            
            # Load the speech audio
            y, sr = librosa.load(temp_filename, sr=None)
            if sample_rate is None:
                sample_rate = sr  # Assume all files have the same sampling rate
            
            # Store the maximum length of the current line's audio across singers
            max_length_line = 0
            singers_shifted = {}
            
            # Apply pitch shifts for each singer
            for singer in ['singer1', 'singer2', 'singer3']:
                n_steps = singer_pitch_shifts[singer][i]
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                singers_shifted[singer] = y_shifted
                if len(y_shifted) > max_length_line:
                    max_length_line = len(y_shifted)
            
            # Pad all singers' audio for the current line to the maximum length
            for singer in ['singer1', 'singer2', 'singer3']:
                y_shifted = singers_shifted[singer]
                if len(y_shifted) < max_length_line:
                    y_shifted = np.pad(y_shifted, (0, max_length_line - len(y_shifted)), 'constant')
                singers_shifted[singer] = y_shifted
                audio_data[singer].append(y_shifted)
            
            # Mix the three singers for the current line
            mixed_line = singers_shifted['singer1'] + singers_shifted['singer2'] + singers_shifted['singer3']
            # Normalize the mixed line to prevent clipping
            max_val = np.max(np.abs(mixed_line))
            if max_val > 0:
                mixed_line = mixed_line / max_val
            mixed_audio_segments.append(mixed_line)
            
            # Remove the temporary file
            os.remove(temp_filename)
        
        # Concatenate all mixed line segments to form the final chorus
        combined_mixed_audio = np.concatenate(mixed_audio_segments)
        
        # Save the mixed chorus audio
        mixed_filename = f'{output_prefix}_chorus.wav'
        sf.write(mixed_filename, combined_mixed_audio, sample_rate)
        print(f"Saved {mixed_filename}")
        
        # Additionally, save individual singers' audio if needed
        for singer in ['singer1', 'singer2', 'singer3']:
            combined_audio = np.concatenate(audio_data[singer])
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = combined_audio / max_val
            # Save the singer's audio file
            singer_filename = f'{output_prefix}_{singer}.wav'
            sf.write(singer_filename, combined_audio, sample_rate)
            print(f"Saved {singer_filename}")
        
        print("Procedural song with chorus generated successfully.")
    
    finally:
        # Clean up temporary directory and all its contents
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")

# Example usage:
if __name__ == "__main__":
    lyrics = """Twinkle twinkle little star
How I wonder what you are
Up above the world so high
Like a diamond in the sky"""
    
    generate_procedural_song_with_chorus(lyrics)
