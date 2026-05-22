"""Data contoh dan loader dataset publik.

File ini bukan inti model. Isinya hanya sumber data agar notebook bisa langsung
dijalankan. Untuk eksperimen yang lebih bagus di Colab, gunakan
`load_talpco_pairs_from_github()` atau `load_csv_pairs()`.
"""

import csv
from urllib.request import urlopen

from .common import normalize_text

TEXT_CORPUS = """
natural language processing studies how computers understand human language.
recurrent neural networks read a sentence one step at a time and keep a hidden
state for previous context. a simple rnn can learn short patterns, but long
sentences are difficult because gradients may vanish during training. lstm
models use input gates, forget gates, and output gates to decide what should be
remembered. this makes lstm useful for text generation, translation, and other
sequence modelling tasks. attention improves encoder decoder models by letting
the decoder focus on important source words while producing each target word.
in this lab we compare simple rnn and lstm language models, then build an
english to indonesian sequence to sequence model with bahdanau attention.
"""

SMALL_TRANSLATION_PAIRS: list[tuple[str, str]] = [
    ("hello", "halo"),
    ("good morning", "selamat pagi"),
    ("how are you", "apa kabar"),
    ("i am fine", "saya baik"),
    ("thank you", "terima kasih"),
    ("i am a student", "saya seorang mahasiswa"),
    ("she is reading a book", "dia sedang membaca buku"),
    ("we are learning language", "kami sedang belajar bahasa"),
    ("the sky is blue", "langit itu biru"),
    ("this book is interesting", "buku ini menarik"),
    ("i live in jakarta", "saya tinggal di jakarta"),
    ("can you help me", "bisakah kamu membantu saya"),
    ("attention shows important words", "perhatian menunjukkan kata penting"),
]


def load_talpco_pairs_from_github(max_samples: int = 1000) -> list[tuple[str, str]]:
    """Ambil English-Indonesian pairs dari GitHub TALPCo.

    Sumber:
    https://github.com/matbahasa/TALPCo

    File English dan Indonesian tersimpan sebagai raw text dengan urutan baris
    paralel. Jadi baris ke-n di English adalah pasangan baris ke-n di
    Indonesian.
    """

    english_url = "https://raw.githubusercontent.com/matbahasa/TALPCo/master/eng/data_eng.txt"
    indonesian_url = "https://raw.githubusercontent.com/matbahasa/TALPCo/master/ind/data_ind.txt"

    english_lines = _download_lines(english_url)
    indonesian_lines = _download_lines(indonesian_url)

    pairs: list[tuple[str, str]] = []
    for english, indonesian in zip(english_lines, indonesian_lines):
        english = normalize_text(english)
        indonesian = normalize_text(indonesian)
        if english and indonesian:
            pairs.append((english, indonesian))
        if len(pairs) >= max_samples:
            break
    return pairs


def load_csv_pairs(
    csv_path: str,
    source_column: str = "english",
    target_column: str = "indonesia",
    max_samples: int | None = None,
    delimiter: str = ",",
) -> list[tuple[str, str]]:
    """Load dataset CSV dari Kaggle/Mendeley/GitHub.

    Contoh:
    `load_csv_pairs("/content/inmad.csv", "english", "indonesia")`
    """

    pairs: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            source = normalize_text(row[source_column])
            target = normalize_text(row[target_column])
            if source and target:
                pairs.append((source, target))
            if max_samples is not None and len(pairs) >= max_samples:
                break
    return pairs


def _download_lines(url: str) -> list[str]:
    with urlopen(url) as response:
        return response.read().decode("utf-8").splitlines()
