"""Task 1: Simple RNN vs LSTM text generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def build_char_vocab(text: str):
    id_to_char = sorted(set(text))
    char_to_id = {char: index for index, char in enumerate(id_to_char)}
    return char_to_id, id_to_char


def encode_chars(text: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[char] for char in text if char in char_to_id]


def decode_chars(ids: list[int], id_to_char: list[str]) -> str:
    return "".join(id_to_char[int(index)] for index in ids)


def make_text_dataloader(text: str, char_to_id: dict[str, int], sequence_length: int = 60, batch_size: int = 32):
    token_ids = torch.tensor(encode_chars(text, char_to_id), dtype=torch.long)
    inputs = []
    targets = []

    for start in range(len(token_ids) - sequence_length):
        inputs.append(token_ids[start : start + sequence_length])
        targets.append(token_ids[start + 1 : start + sequence_length + 1])

    dataset = TensorDataset(torch.stack(inputs), torch.stack(targets))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleRNNTextGenerator(nn.Module):
    """Baseline pembanding memakai nn.RNN."""

    def __init__(self, vocab_size: int, embedding_size: int = 64, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden_state=None):
        embedded = self.embedding(input_ids)
        rnn_output, hidden_state = self.rnn(embedded, hidden_state)
        logits = self.output_layer(rnn_output)
        return logits, hidden_state


class LSTMTextGenerator(nn.Module):
    """Model utama Task 1 memakai nn.LSTM."""

    def __init__(self, vocab_size: int, embedding_size: int = 64, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden_state=None):
        embedded = self.embedding(input_ids)
        lstm_output, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.output_layer(lstm_output)
        return logits, hidden_state


def train_text_generator(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device | str,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = loss_function(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

        losses.append(total_loss / total_tokens)

    return losses


@torch.no_grad()
def generate_text(
    model: nn.Module,
    seed_text: str,
    char_to_id: dict[str, int],
    id_to_char: list[str],
    generated_length: int = 150,
    temperature: float = 0.8,
    device: torch.device | str = "cpu",
) -> str:
    model.eval()

    generated_ids = encode_chars(seed_text, char_to_id)
    if not generated_ids:
        generated_ids = [0]

    input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
    logits, hidden_state = model(input_ids)

    for _ in range(generated_length):
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probabilities = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1).item()

        generated_ids.append(next_id)
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden_state = model(input_ids, hidden_state)

    return decode_chars(generated_ids, id_to_char)
