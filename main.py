import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datasets


# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text


# Dataset do trenowania modelu
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long),
            torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        )


# Model LSTM
class CustomLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.i2f = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.i2i = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.i2g = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, -1)  # Zmieniamy wymiar tensora wejściowego
        combined = torch.cat((embedded, hidden), 1)
        f_t = torch.sigmoid(self.i2f(combined))
        i_t = torch.sigmoid(self.i2i(combined))
        o_t = torch.sigmoid(self.i2o(combined))
        g_t = torch.tanh(self.i2g(combined))
        cell = f_t * cell + i_t * g_t
        hidden = o_t * torch.tanh(cell)
        output = self.h2o(hidden)
        return output, hidden, cell

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)


# Funkcja treningowa
def train_step(model, input_tensor, target_tensor, criterion, optimizer, device):
    hidden, cell = model.init_hidden()
    hidden, cell = hidden.to(device), cell.to(device)
    model.zero_grad()
    loss = 0
    for i in range(len(input_tensor)):
        input_char = input_tensor[i].unsqueeze(0).to(
            device)  # Zmieniamy wymiar tensorów wejściowych i przenosimy do GPU
        target_char = target_tensor[i].to(device)
        output, hidden, cell = model(input_char, hidden, cell)
        loss += criterion(output, target_char.view(1))
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataloader, epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}/{epochs}")
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = train_step(model, inputs, targets, criterion, optimizer, device)
            total_loss += loss
            tqdm.write(f"Batch loss: {loss:.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")


# Funkcja do generowania tekstu
def generate(model, start_str, length, char_to_idx, idx_to_char, device):
    model.eval()
    hidden, cell = model.init_hidden()
    hidden, cell = hidden.to(device), cell.to(device)
    input = torch.tensor([char_to_idx[char] for char in start_str], dtype=torch.long).to(device)
    generated_str = start_str

    for _ in range(length):
        input_char = model.embedding(input[-1].unsqueeze(0))
        output, hidden, cell = model(input_char, hidden, cell)
        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = idx_to_char[top_i.item()]
        generated_str += predicted_char
        input = torch.cat([input, torch.tensor([top_i], dtype=torch.long).to(device)], dim=0)

    return generated_str


# Główna funkcja
if __name__ == "__main__":
    # Sprawdzenie dostępności GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ładowanie danych z WikiText-103
    print("Loading WikiText-103 dataset...")
    dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')

    # Połączenie wszystkich tekstów w jeden ciąg znaków
    print("Combining text from all splits...")
    raw_text = ''
    for split in ['train', 'validation', 'test']:
        raw_text += ' '.join(dataset[split]['text'])

    # Przetwarzanie tekstu
    print("Preprocessing text...")
    processed_text = preprocess_text(raw_text)
    chars = sorted(list(set(processed_text)))
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}
    data = [char_to_idx[char] for char in processed_text]

    # Debug printy
    print(f"Total characters: {len(processed_text)}")
    print(f"Unique characters: {len(chars)}")
    print(f"Sample characters to index mapping: {list(char_to_idx.items())[:10]}")

    # Przygotowanie datasetu
    seq_length = 100  # Długość sekwencji
    print(f"Creating dataset with sequence length: {seq_length}")
    dataset = TextDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Parametry modelu
    vocab_size = len(char_to_idx)
    hidden_size = 128
    output_size = vocab_size

    print("Initializing model...")
    model = CustomLSTM(vocab_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Trening modelu
    print("Starting training...")
    train(model, dataloader, epochs=100, criterion=criterion, optimizer=optimizer, device=device)

    # Generowanie tekstu
    print("Generating text...")
    start_str = "hello"
    generated_text = generate(model, start_str, 100, char_to_idx, idx_to_char, device)
    print("Generated text:")
    print(generated_text)
