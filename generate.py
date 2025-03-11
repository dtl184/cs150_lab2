import torch
import numpy as np
from torch.nn.functional import softmax
from music21 import stream, note, chord, metadata

chord_classes = np.load("chord_classes.npy", allow_pickle=True)
note_classes = np.load("note_classes.npy", allow_pickle=True)
duration_classes = np.load("duration_classes.npy", allow_pickle=True)

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

# Load model
vocab_size = len(chord_classes)
note_output_size = len(note_classes)
duration_output_size = len(duration_classes)

model = JazzMelodyLSTM(vocab_size, note_output_size, duration_output_size)
model.load_state_dict(torch.load("charlie_parker_melody_with_durations_model.pth"))
model.eval()

score = stream.Score()

# Define F Blues chord progression for 12 bars
f_blues_chords = [
    "C7", "F7", "C7", "C7",
    "F7", "F7", "C7", "G7",
    "D-7", "G7", "C7", "C7"
]

chords = {
    'C7': ['C','E','G','Bb'],
    'F7': ['F','A','C','Eb'],
    'G7': ['G','B','D','F'],
    'D-7': ['D','F','A','C']
}

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

def generate_melody(chord_sequence, measure_length=4):
    melody = []

    for chord_symbol in chord_sequence:
        chord_idx = np.where(chord_classes == chord_symbol)[0][0]
        chord_tensor = torch.tensor([[chord_idx]], dtype=torch.long)

        time_in_measure = 0
        while time_in_measure < measure_length:
            with torch.no_grad():
                note_out, duration_out = model(chord_tensor)

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

            if time_in_measure + duration_value > measure_length:
                duration_value = measure_length - time_in_measure

            melody.append((int(note_classes[predicted_note]), duration_value))

            time_in_measure += duration_value

    return melody

def save_to_musicxml(melody, chord_sequence, output_file="generated_f_blues_with_chords.musicxml"):
    melody_stream = stream.Part()
    chord_stream = create_chord_stream(chord_sequence)

    melody_stream.append(metadata.Metadata())
    melody_stream.metadata.title = "Generated 12 Bar F Blues Melody with Chords"
    melody_stream.metadata.composer = "Charlie Parker AI"

    measure_length = 4
    measure_count = 0
    current_melody_measure = stream.Measure(number=measure_count + 1)

    time_in_measure = 0

    for midi_pitch, duration in melody:
        if time_in_measure + duration > measure_length:
            melody_stream.append(current_melody_measure)
            measure_count += 1
            current_melody_measure = stream.Measure(number=measure_count + 1)
            time_in_measure = 0

        n = note.Note(midi_pitch)
        n.quarterLength = duration
        current_melody_measure.append(n)

        time_in_measure += duration

    melody_stream.append(current_melody_measure)

    score.append(chord_stream)
    score.append(melody_stream)

    score.write("musicxml", fp=output_file)
    score.show()
    print(f"MusicXML saved as {output_file}")

# Generate and save melody with chords
generated_melody = generate_melody(f_blues_chords)
save_to_musicxml(generated_melody, f_blues_chords)