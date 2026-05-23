# SequenceModellingNLP_Part2

Implementasi Lab Assignment 2 untuk materi sequence modelling NLP:

- Task 1: Simple RNN vs LSTM character-level text generator.
- Task 2: Encoder-decoder LSTM untuk English-Indonesian translation memakai TALPCo.
- Task 3: Bahdanau additive attention dan visualisasi attention weights.
- Task 4: catatan report dan analisis bisa dibuat terakhir dari angka notebook.

## Struktur

- `src/task1_text_generator.py`: kode Task 1, Simple RNN vs LSTM text generator.
- `src/task2_seq2seq.py`: kode wajib Task 2 dan Task 3.
- `src/optional_task.py`: kode bonus beam search dan pointer-generator.
- `src/common.py`: helper umum seperti seed, vocabulary, BLEU, perplexity.
- `src/sample_data.py`: data contoh untuk Task 1.
- `main.ipynb`: notebook utama untuk training, evaluasi, dan visualisasi.
- `requirements.txt`: dependency minimal untuk local environment.

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
- BLEU sederhana untuk encoder-decoder translation tanpa attention dan dengan attention,
- heatmap attention untuk 3 sample sentences.
- bonus beam search decoding,
- bonus pointer-generator network.

Setup cell di notebook otomatis skip kalau dijalankan lokal, jadi folder `src` lokal tidak tertimpa. Di Google Colab, cell tersebut mengambil `src` dari GitHub dan install `sacrebleu`.

## Dataset Publik

Notebook memakai dataset:

1. TALPCo dari GitHub: `matbahasa/TALPCo`.
2. Dataset di-load langsung di `main.ipynb`, bukan di file Python `src`.
3. File English dan Indonesian dibaca dari raw GitHub, lalu angka ID di awal kalimat dibersihkan.

## Catatan Starter Code

Bagian `src/task2_seq2seq.py` mengikuti starter code dari PDF dosen:

- `Encoder`: embedding + bidirectional `nn.LSTM` + `pack_padded_sequence`.
- `Attention`: additive/Bahdanau attention.
- `Decoder`: embedding + LSTM decoder + attention + output layer.

Beberapa bagian ditambahkan agar bisa jalan end-to-end di notebook: dataloader, padding mask, teacher forcing loop, greedy decoding, dan helper evaluasi.

## Catatan Eksperimen

Untuk hasil report yang lebih kuat, naikkan `max_samples` pada loader TALPCo di `main.ipynb` jika runtime Colab masih kuat.
