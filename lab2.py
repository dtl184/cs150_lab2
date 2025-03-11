import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from music21 import converter, harmony, note

directory = "Omnibook_xml/"

all_chords = []
all_notes = []

for filename in os.listdir(directory):
    if filename.endswith(".xml") or filename.endswith(".musicxml"):
        print(f"Processing {filename}...")
        file_path = os.path.join(directory, filename)
        score = converter.parse(file_path)

        for measure in score.parts[0].getElementsByClass("Measure"):
            for element in measure.getElementsByClass("Harmony"):
                if isinstance(element, harmony.ChordSymbol):
                    all_chords.append((element.figure, element.offset, element.quarterLength))

            for element in measure.notes:
                if isinstance(element, note.Note):
                    all_notes.append((element.pitch.midi, element.offset, element.quarterLength))

# Convert to DataFrames
chords_df = pd.DataFrame(all_chords, columns=["Chord", "Offset", "Duration"])
notes_df = pd.DataFrame(all_notes, columns=["Note (MIDI)", "Offset", "Duration"])

# Save extracted data for inspection
chords_df.to_csv("all_chords.csv", index=False)
notes_df.to_csv("all_melody.csv", index=False)

chord_encoder = LabelEncoder()
chords_df["ChordIndex"] = chord_encoder.fit_transform(chords_df["Chord"])

note_encoder = LabelEncoder()
duration_encoder = LabelEncoder()
notes_df["NoteIndex"] = note_encoder.fit_transform(notes_df["Note (MIDI)"])
notes_df["DurationIndex"] = duration_encoder.fit_transform(notes_df["Duration"])

# Align chords and notes based on their offsets and durations
X = []
Y = []

for _, chord_row in chords_df.iterrows():
    chord_start = chord_row["Offset"]
    chord_end = chord_start + chord_row["Duration"]
    chord_idx = chord_row["ChordIndex"]

    # Select notes that fall within the chord duration
    relevant_notes = notes_df[(notes_df["Offset"] >= chord_start) & (notes_df["Offset"] < chord_end)]

    for _, note_row in relevant_notes.iterrows():
        X.append([chord_idx])
        Y.append((note_row["NoteIndex"], note_row["DurationIndex"]))

X = np.array(X)
Y = np.array(Y)

X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# Create PyTorch Dataset
class JazzDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Create DataLoader
batch_size = 32
dataset = JazzDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        loss_note = criterion(note_out.view(-1, note_output_size), batch_Y[:, 0].reshape(-1))
        loss_duration = criterion(duration_out.view(-1, duration_output_size), batch_Y[:, 1].reshape(-1))
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

print("Model trained on all files and saved successfully!")