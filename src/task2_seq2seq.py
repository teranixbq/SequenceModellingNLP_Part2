import random
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

from .common import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN, build_vocab, decode_ids, encode_tokens, tokenize


def split_pairs(sentence_pairs: Sequence[tuple[str, str]], train_ratio: float = 0.85, seed: int = 42):
    pairs = list(sentence_pairs)
    random.Random(seed).shuffle(pairs)
    split_at = max(1, int(len(pairs) * train_ratio))
    return pairs[:split_at], pairs[split_at:]


def build_translation_data(sentence_pairs: Sequence[tuple[str, str]], train_ratio: float = 0.85, seed: int = 42):
    train_pairs, test_pairs = split_pairs(sentence_pairs, train_ratio, seed)
    source_token_to_id, _ = build_vocab([tokenize(source) for source, _ in train_pairs])
    target_token_to_id, target_id_to_token = build_vocab([tokenize(target) for _, target in train_pairs])

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "source_token_to_id": source_token_to_id,
        "target_token_to_id": target_token_to_id,
        "target_id_to_token": target_id_to_token,
        "copy_map": build_copy_map(source_token_to_id, target_token_to_id),
    }


def build_copy_map(source_token_to_id: dict[str, int], target_token_to_id: dict[str, int]) -> torch.Tensor:
    copy_map = torch.full((len(source_token_to_id),), -1, dtype=torch.long)
    special_tokens = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN}
    for token, source_id in source_token_to_id.items():
        if token not in special_tokens and token in target_token_to_id:
            copy_map[source_id] = target_token_to_id[token]
    return copy_map


def make_translation_dataloader(
    sentence_pairs: Sequence[tuple[str, str]],
    source_token_to_id: dict[str, int],
    target_token_to_id: dict[str, int],
    batch_size: int = 16,
    shuffle: bool = True,
):
    collate_fn = partial(
        collate_translation_batch,
        source_token_to_id=source_token_to_id,
        target_token_to_id=target_token_to_id,
    )
    return DataLoader(list(sentence_pairs), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def collate_translation_batch(batch, source_token_to_id: dict[str, int], target_token_to_id: dict[str, int]):
    source_tensors = []
    target_tensors = []
    source_lengths = []

    for source_text, target_text in batch:
        source_ids = encode_tokens(tokenize(source_text), source_token_to_id, add_eos=True)
        target_ids = encode_tokens(tokenize(target_text), target_token_to_id, add_sos=True, add_eos=True)
        source_tensors.append(torch.tensor(source_ids))
        target_tensors.append(torch.tensor(target_ids))
        source_lengths.append(len(source_ids))

    sources = pad_sequence(source_tensors, batch_first=True, padding_value=source_token_to_id[PAD_TOKEN])
    targets = pad_sequence(target_tensors, batch_first=True, padding_value=target_token_to_id[PAD_TOKEN])
    return sources, torch.tensor(source_lengths), targets


class Encoder(nn.Module):
    """Bidirectional LSTM encoder, sesuai requirement PDF."""

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, pad_id: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, source_ids: torch.Tensor, source_lengths: torch.Tensor):
        embedded = self.embedding(source_ids)
        packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=source_ids.size(1))
        return encoder_outputs, hidden, cell


class Attention(nn.Module):
    """Additive/Bahdanau attention, adaptasi dari starter code dosen."""

    def __init__(self, encoder_output_size: int, decoder_hidden_size: int) -> None:
        super().__init__()
        self.energy_layer = nn.Linear(encoder_output_size + decoder_hidden_size, decoder_hidden_size)
        self.score_layer = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, source_mask: torch.Tensor):
        source_length = encoder_outputs.size(1)
        repeated_hidden = decoder_hidden.unsqueeze(1).repeat(1, source_length, 1)
        energy = torch.tanh(self.energy_layer(torch.cat([repeated_hidden, encoder_outputs], dim=-1)))
        scores = self.score_layer(energy).squeeze(-1)
        scores = scores.masked_fill(~source_mask, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    """Decoder LSTM, bisa tanpa attention atau dengan attention."""

    use_pointer = False
    returns_log_prob = False

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, pad_id: int, use_attention: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        encoder_output_size = hidden_size * 2
        lstm_input_size = embedding_size + encoder_output_size if use_attention else embedding_size
        output_input_size = embedding_size + hidden_size + encoder_output_size if use_attention else hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.attention = Attention(encoder_output_size, hidden_size) if use_attention else None
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(output_input_size, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs, source_mask, source_copy_ids=None):
        embedded = self.embedding(input_token.unsqueeze(1))

        if self.use_attention:
            context, attention_weights = self.attention(hidden[-1], encoder_outputs, source_mask)
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            context = None
            attention_weights = None
            lstm_input = embedded

        decoder_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        if self.use_attention:
            output_input = torch.cat([embedded.squeeze(1), decoder_output.squeeze(1), context], dim=-1)
        else:
            output_input = decoder_output.squeeze(1)

        logits = self.output_layer(output_input)
        return logits, hidden, cell, attention_weights


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, source_pad_id, target_sos_id, target_eos_id, copy_map=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_id = source_pad_id
        self.target_sos_id = target_sos_id
        self.target_eos_id = target_eos_id
        self.hidden_bridge = nn.Linear(encoder.hidden_size * 2, decoder.hidden_size)
        self.cell_bridge = nn.Linear(encoder.hidden_size * 2, decoder.hidden_size)
        self.register_buffer("copy_map", torch.empty(0, dtype=torch.long) if copy_map is None else copy_map.long())

    @property
    def returns_log_prob(self) -> bool:
        return bool(getattr(self.decoder, "returns_log_prob", False))

    def prepare_encoder_result(self, source_ids, source_lengths):
        encoder_outputs, hidden, cell = self.encoder(source_ids, source_lengths)
        hidden = torch.tanh(self.hidden_bridge(torch.cat([hidden[-2], hidden[-1]], dim=-1))).unsqueeze(0)
        cell = torch.tanh(self.cell_bridge(torch.cat([cell[-2], cell[-1]], dim=-1))).unsqueeze(0)
        source_mask = source_ids.ne(self.source_pad_id)
        source_copy_ids = self.copy_map[source_ids] if getattr(self.decoder, "use_pointer", False) else None
        return encoder_outputs, hidden, cell, source_mask, source_copy_ids

    def forward(self, source_ids, source_lengths, target_ids, teacher_forcing_ratio: float = 0.5):
        encoder_outputs, hidden, cell, source_mask, source_copy_ids = self.prepare_encoder_result(source_ids, source_lengths)
        input_token = target_ids[:, 0]
        outputs = []
        attentions = []

        for timestep in range(1, target_ids.size(1)):
            output, hidden, cell, attention_weights = self.decoder(input_token, hidden, cell, encoder_outputs, source_mask, source_copy_ids)
            outputs.append(output.unsqueeze(1))
            if attention_weights is not None:
                attentions.append(attention_weights.unsqueeze(1))
            input_token = target_ids[:, timestep] if random.random() < teacher_forcing_ratio else output.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)
        attentions = torch.cat(attentions, dim=1) if attentions else None
        return outputs, attentions

    @torch.no_grad()
    def greedy_decode(self, source_ids, source_lengths, max_length: int = 20):
        encoder_outputs, hidden, cell, source_mask, source_copy_ids = self.prepare_encoder_result(source_ids, source_lengths)
        input_token = torch.full((source_ids.size(0),), self.target_sos_id, dtype=torch.long, device=source_ids.device)
        decoded_ids = [[] for _ in range(source_ids.size(0))]
        attentions = []
        finished = torch.zeros(source_ids.size(0), dtype=torch.bool, device=source_ids.device)

        for _ in range(max_length):
            output, hidden, cell, attention_weights = self.decoder(input_token, hidden, cell, encoder_outputs, source_mask, source_copy_ids)
            next_token = output.argmax(dim=-1)
            for row, token_id in enumerate(next_token.tolist()):
                if not finished[row]:
                    decoded_ids[row].append(token_id)
            if attention_weights is not None:
                attentions.append(attention_weights.unsqueeze(1))
            finished = finished.logical_or(next_token.eq(self.target_eos_id))
            input_token = next_token
            if finished.all():
                break

        attentions = torch.cat(attentions, dim=1) if attentions else None
        return decoded_ids, attentions


def build_seq2seq_model(source_token_to_id, target_token_to_id, embedding_size=64, hidden_size=128, decoder_type="attention"):
    if decoder_type not in {"basic", "attention"}:
        raise ValueError("decoder_type must be 'basic' or 'attention'. Use src.optional_task for pointer-generator.")

    use_attention = decoder_type == "attention"
    encoder = Encoder(len(source_token_to_id), embedding_size, hidden_size, source_token_to_id[PAD_TOKEN])
    decoder = Decoder(len(target_token_to_id), embedding_size, hidden_size, target_token_to_id[PAD_TOKEN], use_attention)
    return Seq2SeqModel(
        encoder,
        decoder,
        source_pad_id=source_token_to_id[PAD_TOKEN],
        target_sos_id=target_token_to_id[SOS_TOKEN],
        target_eos_id=target_token_to_id[EOS_TOKEN],
    )


def train_seq2seq(model, dataloader, epochs, learning_rate, target_pad_id, device, teacher_forcing_ratio=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss(ignore_index=target_pad_id) if model.returns_log_prob else nn.CrossEntropyLoss(ignore_index=target_pad_id)
    losses = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        for source_ids, source_lengths, target_ids in dataloader:
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            output, _ = model(source_ids, source_lengths, target_ids, teacher_forcing_ratio)
            target_without_sos = target_ids[:, 1:]
            loss = loss_function(output.reshape(-1, output.size(-1)), target_without_sos.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            token_count = target_without_sos.ne(target_pad_id).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
        losses.append(total_loss / max(1, total_tokens))

    return losses


@torch.no_grad()
def translate_sentence(model, sentence, source_token_to_id, target_id_to_token, device, max_length=20):
    source_tokens = tokenize(sentence)
    source_ids = encode_tokens(source_tokens, source_token_to_id, add_eos=True)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)
    predicted_batch, attention_batch = model.greedy_decode(source_tensor, source_lengths, max_length)
    attention = attention_batch[0].cpu() if attention_batch is not None else None
    predicted_tokens = decode_ids(predicted_batch[0], target_id_to_token)
    return {
        "source_tokens": source_tokens + [EOS_TOKEN],
        "predicted_tokens": predicted_tokens,
        "translation": " ".join(predicted_tokens),
        "attention": attention,
    }
