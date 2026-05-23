# SequenceModellingNLP_Part2

Implementasi Lab Assignment 2 untuk materi sequence modelling NLP:

- Task 1: Simple RNN vs LSTM character-level text generator.
- Task 2: Encoder-decoder LSTM untuk English-Indonesian translation.
- Task 3: Bahdanau additive attention dan visualisasi attention weights.
- Task 4: catatan report dan analisis bisa dibuat terakhir dari angka notebook.

## Struktur

- `src/task1_text_generator.py`: kode Task 1, Simple RNN vs LSTM text generator.
- `src/task2_seq2seq.py`: kode wajib Task 2 dan Task 3.
- `src/optional_task.py`: kode bonus beam search dan pointer-generator.
- `src/common.py`: helper umum seperti seed, vocabulary, BLEU, perplexity.
- `src/sample_data.py`: data contoh dan loader dataset publik.
- `main.ipynb`: notebook utama untuk training, evaluasi, dan visualisasi.
- `requirements.txt`: dependency Python.

## Setup

PyTorch lebih aman dijalankan dengan Python 3.11.

```bash
make venv
source venv/bin/activate
make install
```

## Menjalankan Notebook

Buka `main.ipynb` melalui Google Colab, Jupyter dari conda, VS Code, atau notebook runner lain yang sudah tersedia di environment kamu.

Notebook akan menampilkan:

- generated text minimal 100 karakter dari Simple RNN dan LSTM,
- perplexity untuk perbandingan Simple RNN vs LSTM,
- BLEU sederhana untuk encoder-decoder tanpa attention dan dengan attention,
- heatmap attention untuk 3 sample sentences.
- bonus beam search decoding,
- bonus pointer-generator network.

## Dataset Publik

Notebook menyediakan 3 opsi:

1. `load_talpco_pairs_from_github()` untuk mengambil English-Indonesian dari GitHub TALPCo.
2. `SMALL_TRANSLATION_PAIRS` hanya fallback/debug cepat.
3. `load_csv_pairs()` untuk CSV dari Kaggle/Mendeley, misalnya dataset dengan kolom `english` dan `indonesia`.

Default di notebook adalah TALPCo dari GitHub, bukan Hugging Face, karena tidak perlu login Kaggle dan tidak perlu library `datasets`.

## Catatan Starter Code

Bagian `src/task2_seq2seq.py` mengikuti starter code dari PDF dosen:

- `Encoder`: embedding + bidirectional `nn.LSTM` + `pack_padded_sequence`.
- `Attention`: additive/Bahdanau attention.
- `Decoder`: embedding + LSTM decoder + attention + output layer.

Beberapa bagian ditambahkan agar bisa jalan end-to-end di notebook: dataloader, padding mask, teacher forcing loop, greedy decoding, dan helper evaluasi.

## Catatan Eksperimen

Dataset kecil bawaan hanya untuk debug cepat. Untuk hasil report yang lebih kuat, pakai `load_talpco_pairs_from_github()` di Colab dan naikkan jumlah epoch di `main.ipynb`.
