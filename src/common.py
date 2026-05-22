"""Helper umum agar kode notebook tetap pendek dan mudah dibaca."""

import math
import random
import re
from collections import Counter
from typing import Sequence

import torch

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9?.!,']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return text.split() if text else []


def build_vocab(token_lists: Sequence[Sequence[str]], min_freq: int = 1):
    """Return dua object simpel: token_to_id dict dan id_to_token list."""

    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    id_to_token = list(SPECIAL_TOKENS)
    for token, freq in sorted(counter.items()):
        if freq >= min_freq and token not in SPECIAL_TOKENS:
            id_to_token.append(token)

    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_tokens(tokens: Sequence[str], token_to_id: dict[str, int], add_sos: bool = False, add_eos: bool = False):
    ids: list[int] = []
    if add_sos:
        ids.append(token_to_id[SOS_TOKEN])
    ids.extend(token_to_id.get(token, token_to_id[UNK_TOKEN]) for token in tokens)
    if add_eos:
        ids.append(token_to_id[EOS_TOKEN])
    return ids


def decode_ids(ids: Sequence[int], id_to_token: Sequence[str]) -> list[str]:
    special_tokens = set(SPECIAL_TOKENS)
    return [id_to_token[int(index)] for index in ids if id_to_token[int(index)] not in special_tokens]


def perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 20.0))


def calculate_bleu(references: Sequence[str], predictions: Sequence[str]) -> float:
    """BLEU pakai library sacrebleu, jadi tidak perlu implement manual."""

    from sacrebleu.metrics import BLEU

    bleu = BLEU(effective_order=True)
    return float(bleu.corpus_score(list(predictions), [list(references)]).score)
