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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading WikiText-103 dataset...")
dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')

def ensure_unicode(text):
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    elif isinstance(text, str):
        return text
    else:
        raise ValueError("Unsupported string type")

def filter_text(text):
    return ''.join([char for char in text if char in all_chars])

train_text = ' '.join(ensure_unicode(text) for text in dataset['train']['text'])
val_text = ' '.join(ensure_unicode(text) for text in dataset['validation']['text'])
test_text = ' '.join(ensure_unicode(text) for text in dataset['test']['text'])

all_chars = string.printable
number_of_chars = len(all_chars)
print(all_chars)

# Filtrowanie tekstów
train_text = filter_text(train_text)
val_text = filter_text(val_text)
test_text = filter_text(test_text)
all_text = train_text + val_text + test_text
number_of_char = len(all_text)

print(len(all_text))
print(len(train_text))
print(len(test_text))
print(len(val_text))
print(type(train_text))
print(type(val_text))
print(type(test_text))

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
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
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
        for epoch in range(1, self.num_epochs + 1):
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

            writer.add_scalar('Training Loss', loss, global_step=epoch)

gennames = Generator()
gennames.train()
