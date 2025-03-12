import torch
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from music21 import stream, note, chord, metadata, meter, harmony
import markovify
import subprocess
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


# Function to generate a melody
def generate_melody(chord_sequence, model, length=48):
    chord_indices = []
    for c in chord_sequence:
        chord_indices.append(np.where(chord_classes == c[0])[0][0])
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
            if c.split('x')[0] not in chord_classes:
                return None
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
        chords_for_model.append((c.split('x')[0], c.split('x')[1]))
        if '-' in c and 'm' in c:
            c = c.replace('m', '')
        chord_stream.append(chord.Chord(harmony.ChordSymbol(c.split('x')[0]).pitches, quarterLength=int(c.split('x')[1])))
        
    return chord_stream, chords_for_model


# Convert to MusicXML using music21
def generate_score(melody, chord_stream):
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = 'CharlAI'
    score.metadata.composer = 'Daniel Little and Emily Ertle'
    melody_stream = stream.Part()
    time_signature = meter.TimeSignature('4/4')
    score.append(time_signature)


    chord_length = 0
    for c in chord_stream.notes:
        chord_length += c.duration.quarterLength

    flag = False
    current_chord_idx = 0
    melody_length = 0
    measure_pos = 0

    while not flag:
        for midi_pitch, duration in melody:
            # Continue while the melody is not longer than the chords
            if melody_length >= chord_length:
                flag = True
                break
            melody_stream.append(note.Note(midi_pitch, quarterLength=duration))
            
            measure_pos += duration
            melody_length += duration
            if melody_length >= chord_length:
                flag = True
                break
            if measure_pos > 4:
                current_chord_idx += 1
                measure_pos -= 4

    score.append(chord_stream)
    score.append(melody_stream)

    return score
    
    
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

    if '-t' in sys.argv:
        subprocess.run(['python', 'train.py'], check=True, text=True)

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
    generated_melody = generate_melody(model=model, chord_sequence=chords_for_model, length=96)

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