import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import string
import random
import sys
import unicodedata
from torch.utils.tensorboard import SummaryWriter
import datasets
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filepath = "cleaned_text.txt"


def ensure_unicode(text):
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    elif isinstance(text, str):
        return text
    else:
        raise ValueError("Unsupported string type")


# Function to filter text
def filter_text(text, allowed_chars):
    return ''.join([char for char in text if char in allowed_chars])


# Load and process the file
with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
    content = file.read()
    content = ensure_unicode(content)
    all_chars = string.printable
    filtered_text = filter_text(content, all_chars)

# Use filtered text as 'all_text'
all_text = filtered_text
number_of_char = len(all_text)

all_chars = string.printable
number_of_chars = len(all_chars)
print(all_chars)
print(len(all_text))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 10000
        self.batch_size = 1
        self.print_every = 25
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_chars.index(string[c])
        return tensor

    def get_random_batch(self):
        start_index = random.randint(0, len(all_text) - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        text_str = all_text[start_index:end_index]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])
        return text_input.long(), text_target.long()

    def generate(self, initial_string='T', prediction_length=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_string)
        predicted = initial_string

        for p in range(len(initial_string) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        last_char = initial_input[-1]

        for p in range(prediction_length):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_chars[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)
        return predicted

    def train(self):
        self.rnn = RNN(number_of_chars, self.hidden_size, self.num_layers, number_of_chars).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/names0')
        print("=> starting training :)")

        start_time = time.time()
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="Training", unit="epoch"):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f'Loss: {loss}')
                print(self.generate())
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / epoch * self.num_epochs
                print(f'Estimated total training time: {estimated_total_time // 60} minutes')

            writer.add_scalar('Training Loss', loss, global_step=epoch)

        # Save the model after training
        torch.save(self.rnn.state_dict(), 'trained_rnn.pth')
        print("=> training finished and model saved :)")


gennames = Generator()
gennames.train()
