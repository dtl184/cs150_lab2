import torch
import numpy as np
from torch.nn.functional import softmax
from music21 import stream, note, chord, metadata

# Load trained model & encoders
chord_classes = np.load("chord_classes.npy", allow_pickle=True)
note_classes = np.load("note_classes.npy", allow_pickle=True)

# Define the same model structure
class JazzMelodyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim=32, hidden_dim=128):
        super(JazzMelodyLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.softmax(x)

# Load model
vocab_size = len(chord_classes)
output_size = len(note_classes)
model = JazzMelodyLSTM(vocab_size, output_size)
model.load_state_dict(torch.load("charlie_parker_melody_model.pth"))
model.eval()


score = stream.Score()
harmony = stream.Part()

# Define F Blues chord progression
f_blues_chords = [
    "C7", "F7", "C7", "C7",
    "F7", "F7", "C7", "G7",
    "D-7", "G7", "C7", "C7"
]

for c in f_blues_chords:
    if c not in chord_classes:
        print(f"Chord {c} is missing from chord_classes!")

# Function to generate a melody
def generate_melody(chord_sequence, length=48):
    chord_indices = [np.where(chord_classes == c)[0][0] for c in chord_sequence]
    chord_indices = torch.tensor(chord_indices, dtype=torch.long).unsqueeze(0)

    melody_indices = []
    for _ in range(length):
        with torch.no_grad():
            output = model(chord_indices)
        predicted_probs = softmax(output[0, -1], dim=0)
        predicted_index = torch.argmax(predicted_probs).item()
        melody_indices.append(predicted_index)

        # Shift the input window forward
        chord_indices = torch.roll(chord_indices, shifts=-1, dims=1)
        chord_indices[0, -1] = predicted_index  # Use predicted note as input

    # Convert back to MIDI notes
    melody_notes = [int(note_classes[i]) for i in melody_indices]
    return melody_notes

# Generate melody
generated_melody = generate_melody(f_blues_chords, length=48)

# Convert to MusicXML using music21
def save_to_musicxml(midi_notes, chord_sequence, output_file="generated_f_blues.musicxml"):
    melody_stream = stream.Part()
    melody_stream.append(metadata.Metadata())
    melody_stream.metadata.title = "Generated F Blues Melody"
    melody_stream.metadata.composer = "Charlie Parker AI"

    measure_length = 4  # Four beats per measure
    measure_count = 0
    current_measure = stream.Measure(number=measure_count + 1)
    
    for i, midi_pitch in enumerate(midi_notes):
        if i % measure_length == 0:
            #if measure_count < len(chord_sequence):
                #chord_symbol = chord.ChordSymbol(chord_sequence[measure_count])
                #current_measure.insert(0, chord_symbol)
            melody_stream.append(current_measure)
            measure_count += 1
            current_measure = stream.Measure(number=measure_count + 1)
        
        n = note.Note(midi_pitch)
        n.quarterLength = 1  # Quarter note duration
        current_measure.append(n)
    
    melody_stream.append(current_measure)
    melody_stream.write("musicxml", fp=output_file)
    melody_stream.show()
    print(f"MusicXML saved as {output_file}")

# Save melody to MusicXML
save_to_musicxml(generated_melody, f_blues_chords)
