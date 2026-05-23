## Struktur

- `src/task1_text_generator.py`: kode Task 1, Simple RNN vs LSTM text generator.
- `src/task2_seq2seq.py`: kode wajib Task 2 dan Task 3.
- `src/optional_task.py`: kode bonus beam search dan pointer-generator.
- `src/common.py`: helper umum seperti seed, vocabulary, BLEU, perplexity.
- `src/sample_data.py`: data contoh untuk Task 1.
- `main.ipynb`: notebook utama untuk training, evaluasi, dan visualisasi.
- `requirements.txt`: dependency minimal untuk local environment.

## How to Run
### 1. Buka Google Colab

Buka: https://colab.research.google.com

---

## 2. Open Notebook Via Link GitHub

Klik:
```bash
File → Open Notebook → GitHub → Salin link repo ini
```
<img width="989" height="768" alt="image" src="https://github.com/user-attachments/assets/17a9e4bf-c4be-4e88-8f0c-19c1fa4827a4" />


## Dataset Publik

Notebook memakai dataset:

1. TALPCo dari GitHub: `matbahasa/TALPCo`  https://github.com/matbahasa/TALPCo

## Evaluasi

Evaluasi dilakukan langsung di `main.ipynb` setelah training selesai. Metric yang digunakan:

- Perplexity untuk membandingkan Simple RNN dan LSTM pada Task 1.
- BLEU score untuk membandingkan Encoder-Decoder tanpa Attention dan dengan Attention pada Task 2 dan Task 3.
- Attention heatmap untuk melihat kata source yang diperhatikan decoder.
- Beam Search dievaluasi secara kualitatif dengan membandingkan output greedy decoding dan beam decoding.
- Final loss untuk model bonus Pointer-Generator.

Hasil eksperimen terakhir:

| Eksperimen | Metric | Hasil |
|---|---:|---:|
| Simple RNN Text Generator | Perplexity | 1.12 |
| LSTM Text Generator | Perplexity | 1.11 |
| Encoder-Decoder tanpa Attention | BLEU | 7.34 |
| Encoder-Decoder dengan Attention | BLEU | 10.81 |
