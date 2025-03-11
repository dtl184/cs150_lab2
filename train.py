import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from music21 import converter, harmony, note
import markovify
import json

# Create PyTorch Dataset
class JazzDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
# Define the LSTM Model
class JazzMelodyLSTM(nn.Module):
    def __init__(self, vocab_size, note_output_size, duration_output_size, embedding_dim=32, hidden_dim=128):
        super(JazzMelodyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_note = nn.Linear(hidden_dim, note_output_size)
        self.fc_duration = nn.Linear(hidden_dim, duration_output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        note_out = self.softmax(self.fc_note(x))
        duration_out = self.softmax(self.fc_duration(x))
        return note_out, duration_out
    

# Directory containing Charlie Parker MusicXML files
directory = "Omnibook_xml/"

# Lists to store extracted chords and melody notes
all_chords = []
all_notes = []
chords_str = ''
markov_chain = {}

# Loop through all MusicXML files
for filename in os.listdir(directory):
    if filename.endswith(".xml") or filename.endswith(".musicxml"):
        print(f"Processing {filename}...")
        file_path = os.path.join(directory, filename)
        score = converter.parse(file_path)

        # Extract chords and melody notes
        chords = []
        notes = []
        # Also prepare text file for markovify or dict for handmade markov chain
        last_offset = 0
        last_chord = None

        i = 0
        for measure in score.parts[0].getElementsByClass("Measure"):
            for element in measure.getElementsByClass("Harmony"):
                if isinstance(element, harmony.ChordSymbol):
                    chords.append((element.figure, element.offset))
                    if last_offset == 0 and element.offset == 0:
                        chord_length = 4
                    else:
                        chord_length = int(abs(element.offset - last_offset))
                    chords_str += f'{element.figure}x{chord_length} '
                    if last_chord is not None:
                        if last_chord not in markov_chain:
                            markov_chain[last_chord] = []
                        markov_chain[last_chord].append(f'{element.figure}x{int(chord_length)}')
                    last_chord = f'{element.figure}x{chord_length}'
                    last_offset = element.offset

            for element in measure.notes:
                if isinstance(element, note.Note):
                    notes.append((element.pitch.midi, element.offset, element.quarterLength))

        # Append extracted data to global lists
        all_chords.extend(chords)
        all_notes.extend(notes)
        chords_str = chords_str.rstrip(chords_str[-1])
        chords_str += '. '
        

# Convert to DataFrames
chords_df = pd.DataFrame(all_chords, columns=["Chord", "Offset"])
notes_df = pd.DataFrame(all_notes, columns=["Note (MIDI)", "Offset", "Duration"])

with open('dict_markov.json', 'w') as f:
    json.dump(markov_chain, f)
    
markov_model = markovify.Text(chords_str)

model_json = markov_model.to_json()
with open('markovify.json', 'w') as f:
    json.dump(model_json, f)

# Save extracted data for inspection
chords_df.to_csv("all_chords.csv", index=False)
notes_df.to_csv("all_melody.csv", index=False)


# Step 1: Encode Chords as Input (X)
chord_encoder = LabelEncoder()
chords_df["ChordIndex"] = chord_encoder.fit_transform(chords_df["Chord"])

# Step 2: Encode Notes and Durations as Target (Y)
note_encoder = LabelEncoder()
duration_encoder = LabelEncoder()
notes_df["NoteIndex"] = note_encoder.fit_transform(notes_df["Note (MIDI)"])
notes_df["DurationIndex"] = duration_encoder.fit_transform(notes_df["Duration"])

# Step 3: Align Chords & Notes into Sequences
time_steps = 32

X = []
Y = []
for i in range(len(chords_df) - time_steps):
    X.append(chords_df["ChordIndex"].iloc[i: i + time_steps].values)
    Y.append(
        list(
            zip(
                notes_df["NoteIndex"].iloc[i: i + time_steps].values,
                notes_df["DurationIndex"].iloc[i: i + time_steps].values,
            )
        )
    )

X = np.array(X)
Y = np.array(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# Create DataLoader
batch_size = 32
dataset = JazzDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
vocab_size = len(chord_encoder.classes_)
note_output_size = len(note_encoder.classes_)
duration_output_size = len(duration_encoder.classes_)

model = JazzMelodyLSTM(vocab_size, note_output_size, duration_output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        note_out, duration_out = model(batch_X)
        loss_note = criterion(note_out.view(-1, note_output_size), batch_Y[:, :, 0].reshape(-1))
        loss_duration = criterion(duration_out.view(-1, duration_output_size), batch_Y[:, :, 1].reshape(-1))
        loss = loss_note + loss_duration
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save Model & Encoders
torch.save(model.state_dict(), "charlie_parker_melody_with_durations_model.pth")
np.save("chord_classes.npy", chord_encoder.classes_)
np.save("note_classes.npy", note_encoder.classes_)
np.save("duration_classes.npy", duration_encoder.classes_)

print("Models trained on all files and saved successfully!")
