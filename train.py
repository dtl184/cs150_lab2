import os
import cupy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from music21 import converter, harmony, note
import logging

# Setup logging
logging.basicConfig(
    filename='training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

directory = "Omnibook_xml/"

all_chords = []
all_notes = []

# Limit data for faster processing
max_files = 5

# Data extraction
for idx, filename in enumerate(os.listdir(directory)):
    if idx >= max_files:
        break
    if filename.endswith(".xml") or filename.endswith(".musicxml"):
        logging.info(f"Processing {filename}...")
        file_path = os.path.join(directory, filename)
        score = converter.parse(file_path)

        for measure in score.parts[0].getElementsByClass("Measure"):
            for element in measure.getElementsByClass("Harmony"):
                if isinstance(element, harmony.ChordSymbol):
                    duration = element.quarterLength if element.quarterLength > 0 else 1.0
                    all_chords.append((element.figure, element.offset, duration))

            for element in measure.notes:
                if isinstance(element, note.Note):
                    duration = element.quarterLength if element.quarterLength > 0 else 1.0
                    all_notes.append((element.pitch.midi, element.offset, duration))

chords_df = pd.DataFrame(all_chords, columns=["Chord", "Offset", "Duration"])
notes_df = pd.DataFrame(all_notes, columns=["Note (MIDI)", "Offset", "Duration"])

chord_encoder = LabelEncoder()
chords_df["ChordIndex"] = chord_encoder.fit_transform(chords_df["Chord"])

note_encoder = LabelEncoder()
duration_encoder = LabelEncoder()
notes_df["NoteIndex"] = note_encoder.fit_transform(notes_df["Note (MIDI)"])
notes_df["DurationIndex"] = duration_encoder.fit_transform(notes_df["Duration"])

# Reduce dataset size for faster training
chords_df = chords_df.sample(frac=0.5)
notes_df = notes_df.sample(frac=0.5)

# Convert to GPU arrays
chord_offsets = cp.asarray(chords_df["Offset"].values, dtype=cp.float32)
chord_durations = cp.asarray(chords_df["Duration"].values, dtype=cp.float32)
chord_indices = cp.asarray(chords_df["ChordIndex"].values, dtype=cp.int32)

note_offsets = cp.asarray(notes_df["Offset"].values, dtype=cp.float32)
note_indices = cp.asarray(notes_df["NoteIndex"].values, dtype=cp.int32)
duration_indices = cp.asarray(notes_df["DurationIndex"].values, dtype=cp.int32)

X = []
Y = []

for chord_start, chord_duration, chord_idx in zip(chord_offsets, chord_durations, chord_indices):
    chord_end = chord_start + chord_duration

    mask = (note_offsets >= chord_start) & (note_offsets < chord_end)
    relevant_note_indices = note_indices[mask]
    relevant_duration_indices = duration_indices[mask]

    if relevant_note_indices.size > 0:
        chord_input = cp.repeat(chord_idx, relevant_note_indices.size)
        X.extend(cp.asnumpy(chord_input[:, cp.newaxis]))
        Y.extend(zip(cp.asnumpy(relevant_note_indices), cp.asnumpy(relevant_duration_indices)))

X = np.array(X)
Y = np.array(Y)

X_tensor = torch.tensor(X, dtype=torch.long, device=device)
Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)

class JazzDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

batch_size = 64  # Increased batch size for faster training
dataset = JazzDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

vocab_size = len(chord_encoder.classes_)
note_output_size = len(note_encoder.classes_)
duration_output_size = len(duration_encoder.classes_)

model = JazzMelodyLSTM(vocab_size, note_output_size, duration_output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100  # Reduced number of epochs for faster completion
logging.info("Starting model training...")

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

    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "charlie_parker_melody_with_durations_model.pth")
np.save("chord_classes.npy", chord_encoder.classes_)
np.save("note_classes.npy", note_encoder.classes_)
np.save("duration_classes.npy", duration_encoder.classes_)

logging.info("Model trained on all files and saved successfully!")