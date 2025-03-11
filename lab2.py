import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from music21 import converter, harmony, note

# Directory containing Charlie Parker MusicXML files
directory = "Omnibook_xml/"

# Lists to store extracted chords and melody notes
all_chords = []
all_notes = []

# Loop through all MusicXML files
for filename in os.listdir(directory):
    if filename.endswith(".xml") or filename.endswith(".musicxml"):
        print(f"Processing {filename}...")
        file_path = os.path.join(directory, filename)
        score = converter.parse(file_path)

        # Extract chords and melody notes
        chords = []
        notes = []

        for measure in score.parts[0].getElementsByClass("Measure"):
            for element in measure.getElementsByClass("Harmony"):
                if isinstance(element, harmony.ChordSymbol):
                    chords.append((element.figure, element.offset))

            for element in measure.notes:
                if isinstance(element, note.Note):
                    notes.append((element.pitch.midi, element.offset))

        # Append extracted data to global lists
        all_chords.extend(chords)
        all_notes.extend(notes)

# Convert to DataFrames
chords_df = pd.DataFrame(all_chords, columns=["Chord", "Offset"])
notes_df = pd.DataFrame(all_notes, columns=["Note (MIDI)", "Offset"])

# 🔹 Save extracted data for inspection
chords_df.to_csv("all_chords.csv", index=False)
notes_df.to_csv("all_melody.csv", index=False)

# Step 1: Encode Chords as Input (X)
chord_encoder = LabelEncoder()
chords_df["ChordIndex"] = chord_encoder.fit_transform(chords_df["Chord"])

# Step 2: Encode Notes as Target (Y)
note_encoder = LabelEncoder()
notes_df["NoteIndex"] = note_encoder.fit_transform(notes_df["Note (MIDI)"])

# Step 3: Align Chords & Notes into Sequences
time_steps = 32  # 32 sixteenth-note steps per input

X = []
Y = []
for i in range(len(chords_df) - time_steps):
    X.append(chords_df["ChordIndex"].iloc[i : i + time_steps].values)
    Y.append(notes_df["NoteIndex"].iloc[i : i + time_steps].values)

X = np.array(X)
Y = np.array(Y)

# Convert to PyTorch tensors
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
    def __init__(self, vocab_size, output_size, embedding_dim=32, hidden_dim=128):
        super(JazzMelodyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # Apply log softmax for classification

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.softmax(x)

# Initialize Model
vocab_size = len(chord_encoder.classes_)  # Number of unique chords
output_size = len(note_encoder.classes_)  # Number of unique melody notes

model = JazzMelodyLSTM(vocab_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output.view(-1, output_size), batch_Y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save Model & Encoders
torch.save(model.state_dict(), "charlie_parker_melody_model.pth")
np.save("chord_classes.npy", chord_encoder.classes_)
np.save("note_classes.npy", note_encoder.classes_)

print("Model trained on all files and saved successfully!")
