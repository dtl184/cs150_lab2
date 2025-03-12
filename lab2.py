import torch
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from music21 import stream, note, chord, metadata, meter
import markovify
import json
import sys

# Load trained model & encoders
chord_classes = np.load("chord_classes.npy", allow_pickle=True)
note_classes = np.load("note_classes.npy", allow_pickle=True)
duration_classes = np.load("duration_classes.npy", allow_pickle=True)

# Define the same model structure
class JazzMelodyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, note_output_size, duration_output_size, embedding_dim=32, hidden_dim=128):
        super(JazzMelodyLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_note = torch.nn.Linear(hidden_dim, note_output_size)
        self.fc_duration = torch.nn.Linear(hidden_dim, duration_output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)


    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        note_out = self.softmax(self.fc_note(x))
        duration_out = self.softmax(self.fc_duration(x))
        return note_out, duration_out
    

def create_chord_stream(chord_sequence):
    chord_stream = stream.Part()
    measure_count = 1

    for chord_symbol in chord_sequence:
        current_measure = stream.Measure(number=measure_count)
        block_chord = chord.Chord(chords[chord_symbol])
        block_chord.quarterLength = 4.0  # Whole note duration
        current_measure.append(block_chord)
        chord_stream.append(current_measure)
        measure_count += 1

    return chord_stream


# Function to generate a melody
def generate_melody(chord_sequence, model, length=48):
    chord_indices = [np.where(chord_classes == c)[0][0] for c in chord_sequence]
    chord_indices = torch.tensor(chord_indices, dtype=torch.long).unsqueeze(0)

    melody = []
    for _ in range(length):
        with torch.no_grad():
            note_out, duration_out = model(chord_indices)
        
        note_probs = softmax(note_out[0, -1], dim=0)
        duration_probs = softmax(duration_out[0, -1], dim=0)
        
        predicted_note = torch.multinomial(note_probs, 1).item()
        predicted_duration = torch.multinomial(duration_probs, 1).item()

        if np.random.rand() < 0.7:
            duration_value = 0.5
        elif np.random.rand() < 0.85:
            duration_value = 1.0
        else:
            duration_value = 0.25

        melody.append((int(note_classes[predicted_note]), duration_value))

        chord_indices = torch.roll(chord_indices, shifts=-1, dims=1)
        chord_indices[0, -1] = predicted_note

    return melody


def generate_chords(num_bars=24):
    chord_stream = stream.Part()
    # Open and read the JSON files
    with open('markovify.json', 'r') as f:
        model_json = json.load(f)
    
    markov_model = markovify.Text.from_json(model_json)
    
    def chord_length_filter(sentence, num_bars=24):
        desired_chord_length = num_bars * 4
        curr_chord_length = 0
        chords = sentence.split()
        for c in chords:
            c = c.rstrip('.')
            curr_chord_length += int(c.split('x')[1])
            
        
        if curr_chord_length == desired_chord_length:
            print(desired_chord_length, curr_chord_length)
            return sentence
        else:
            return None
    
    sentence = None
    while sentence == False or sentence is None:  # Keep generating until we get a valid one
        sentence = markov_model.make_sentence(tries=100, min_chars=num_bars*4, max_chars=num_bars*11)
        sentence = chord_length_filter(sentence, num_bars)
    
    chords_for_model = []
    chords = sentence.split()
    for c in chords:
        c = c.rstrip('.')
        chord_stream.append(chord.Chord(c.split('x')[0], quarter_length=c.split('x')[1]))
        chords_for_model.append((c.split('x')[0], c.split('x')[1]))
        
    return chord_stream, chords_for_model


# Convert to MusicXML using music21
def generate_score(melody, chord_sequence, output_file="generated_f_blues_with_chords.musicxml"):
    score = stream.Score()
    melody_stream = stream.Part()
    chord_stream = stream.Part()

    melody_stream.append(metadata.Metadata())
    melody_stream.metadata.title = "Generated 12 Bar F Blues Melody with Chords"
    melody_stream.metadata.composer = "Charlie Parker AI"

    measure_length = 4
    measure_count = 0
    current_melody_measure = stream.Measure(number=measure_count + 1)
    current_chord_measure = stream.Measure(number=measure_count + 1)

    time_in_measure = 0
    chord_idx = 0

    for midi_pitch, duration in melody:
        if time_in_measure + duration > measure_length:
            melody_stream.append(current_melody_measure)
            chord_stream.append(current_chord_measure)
            measure_count += 1
            current_melody_measure = stream.Measure(number=measure_count + 1)
            current_chord_measure = stream.Measure(number=measure_count + 1)
            time_in_measure = 0

        n = note.Note(midi_pitch)
        n.quarterLength = duration
        current_melody_measure.append(n)

        time_in_measure += duration

    melody_stream.append(current_melody_measure)
    chord_stream = create_chord_stream()

    score.append(chord_stream)
    score.append(melody_stream)

    score.write("musicxml", fp=output_file)
    score.show()
    
    
# Function to enforce Markovify word count
def chord_length_filter(sentence, num_bars=12):
    desired_chord_length = num_bars * 4
    curr_chord_length = 0
    chords = sentence.split()
    for c in chords:
        c = c.rstrip('.')
        curr_chord_length += int(c.split('x')[1])
    
    if curr_chord_length == desired_chord_length:
        print(desired_chord_length, curr_chord_length)
        return sentence
    else:
        return None

def main():
    # Load model
    vocab_size = len(chord_classes)
    note_output_size = len(note_classes)
    duration_output_size = len(duration_classes)

    model = JazzMelodyLSTM(vocab_size, note_output_size, duration_output_size)
    model.load_state_dict(torch.load("charlie_parker_melody_with_durations_model.pth"))
    model.eval()
        
    """with open('dict_markov.json', 'r') as f:
        markov_dict = json.load(f)"""
    
    num_bars=24
        
    chord_stream, chords_for_model = generate_chords(num_bars)

    # Generate melody
    generated_melody = generate_melody(model, chords_for_model, length=96)

    # Save melody and chords to MusicXML
    score = generate_score(generated_melody, chord_stream)
    

    # Play midi, output sheet music, or print the contents of the stream
    if ("-m" in sys.argv):
        score.show('midi')
    elif ("-s" in sys.argv):
        score.show()
    else:
        score.show('midi')

if __name__ == "__main__":
    main()