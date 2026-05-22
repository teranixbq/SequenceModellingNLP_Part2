"""Bonus: Beam Search dan Pointer-Generator Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN, decode_ids, encode_tokens, tokenize
from .task2_seq2seq import Encoder, Seq2SeqModel


class PointerGeneratorDecoder(nn.Module):
    use_pointer = True
    returns_log_prob = True

    def __init__(self, vocab_size, embedding_size, hidden_size, pad_id, unk_id):
        super().__init__()
        self.hidden_size = hidden_size
        self.unk_id = unk_id
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.attention_layer = nn.Linear(hidden_size * 3, hidden_size)
        self.score_layer = nn.Linear(hidden_size, 1, bias=False)
        self.lstm = nn.LSTM(embedding_size + hidden_size * 2, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(embedding_size + hidden_size * 3, vocab_size)
        self.pointer_gate = nn.Linear(embedding_size + hidden_size * 3, 1)

    def attention(self, hidden, encoder_outputs, source_mask):
        source_length = encoder_outputs.size(1)
        repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, source_length, 1)
        energy = torch.tanh(self.attention_layer(torch.cat([repeated_hidden, encoder_outputs], dim=-1)))
        scores = self.score_layer(energy).squeeze(-1).masked_fill(~source_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

    def forward(self, input_token, hidden, cell, encoder_outputs, source_mask, source_copy_ids):
        embedded = self.embedding(input_token.unsqueeze(1))
        context, attention_weights = self.attention(hidden, encoder_outputs, source_mask)
        decoder_output, (hidden, cell) = self.lstm(torch.cat([embedded, context.unsqueeze(1)], dim=-1), (hidden, cell))
        output_input = torch.cat([embedded.squeeze(1), decoder_output.squeeze(1), context], dim=-1)

        vocab_distribution = F.softmax(self.output_layer(output_input), dim=-1)
        copy_distribution = torch.zeros_like(vocab_distribution)
        valid_copy = source_copy_ids.ge(0)
        copy_distribution.scatter_add_(1, source_copy_ids.clamp_min(0), attention_weights * valid_copy.float())

        unknown_mass = (attention_weights * (~valid_copy).float()).sum(dim=1, keepdim=True)
        unknown_ids = torch.full((input_token.size(0), 1), self.unk_id, dtype=torch.long, device=input_token.device)
        copy_distribution.scatter_add_(1, unknown_ids, unknown_mass)

        p_gen = torch.sigmoid(self.pointer_gate(output_input))
        final_distribution = p_gen * vocab_distribution + (1.0 - p_gen) * copy_distribution
        return torch.log(final_distribution.clamp_min(1e-12)), hidden, cell, attention_weights


def build_pointer_generator_model(source_token_to_id, target_token_to_id, copy_map, embedding_size=64, hidden_size=128):
    encoder = Encoder(len(source_token_to_id), embedding_size, hidden_size, source_token_to_id[PAD_TOKEN])
    decoder = PointerGeneratorDecoder(
        len(target_token_to_id),
        embedding_size,
        hidden_size,
        target_token_to_id[PAD_TOKEN],
        target_token_to_id[UNK_TOKEN],
    )
    return Seq2SeqModel(
        encoder,
        decoder,
        source_pad_id=source_token_to_id[PAD_TOKEN],
        target_sos_id=target_token_to_id[SOS_TOKEN],
        target_eos_id=target_token_to_id[EOS_TOKEN],
        copy_map=copy_map,
    )


@torch.no_grad()
def beam_search_ids(model, source_ids, source_lengths, beam_size=3, max_length=20):
    encoder_outputs, hidden, cell, source_mask, source_copy_ids = model.prepare_encoder_result(source_ids, source_lengths)
    beams = [([], 0.0, hidden, cell, torch.tensor([model.target_sos_id], device=source_ids.device), [])]

    for _ in range(max_length):
        candidates = []
        for token_ids, score, hidden, cell, input_token, attention_list in beams:
            if token_ids and token_ids[-1] == model.target_eos_id:
                candidates.append((token_ids, score, hidden, cell, input_token, attention_list))
                continue

            output, next_hidden, next_cell, attention = model.decoder(
                input_token, hidden, cell, encoder_outputs, source_mask, source_copy_ids
            )
            log_probs = output if model.returns_log_prob else F.log_softmax(output, dim=-1)
            top_scores, top_tokens = torch.topk(log_probs.squeeze(0), beam_size)

            for token_score, token_id in zip(top_scores.tolist(), top_tokens.tolist()):
                next_attention = list(attention_list)
                if attention is not None:
                    next_attention.append(attention.squeeze(0).cpu())
                candidates.append(
                    (
                        token_ids + [token_id],
                        score + float(token_score),
                        next_hidden.clone(),
                        next_cell.clone(),
                        torch.tensor([token_id], device=source_ids.device),
                        next_attention,
                    )
                )

        beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0])), reverse=True)[:beam_size]
        if all(token_ids and token_ids[-1] == model.target_eos_id for token_ids, *_ in beams):
            break

    best_tokens, _, _, _, _, best_attention = beams[0]
    best_attention = torch.stack(best_attention) if best_attention else None
    return best_tokens, best_attention


@torch.no_grad()
def translate_with_beam_search(model, sentence, source_token_to_id, target_id_to_token, device, beam_size=3, max_length=20):
    source_tokens = tokenize(sentence)
    source_ids = encode_tokens(source_tokens, source_token_to_id, add_eos=True)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)
    predicted_ids, attention = beam_search_ids(model, source_tensor, source_lengths, beam_size, max_length)
    predicted_tokens = decode_ids(predicted_ids, target_id_to_token)
    return {
        "source_tokens": source_tokens + [EOS_TOKEN],
        "predicted_tokens": predicted_tokens,
        "translation": " ".join(predicted_tokens),
        "attention": attention,
    }
