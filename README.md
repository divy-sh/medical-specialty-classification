# Medical Specialty Classification

Classifying medical transcriptions by specialty (Cardiology, Neurology, Orthopedics, …) using classical ML, deep learning, and transformer-based models.

This repository is a collection of Jupyter notebooks that explore the problem end to end — from raw transcription text to a trained classifier — and compare a range of approaches: Naive Bayes, Logistic Regression, Random Forest, CNNs, LSTMs/GRUs, BERT embeddings, and a fine-tuned ClinicalBERT.

## Dataset

All notebooks use the [Medical Transcriptions dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) from Kaggle (`mtsamples.csv`): ~5,000 transcribed medical reports labeled with a medical specialty. The dataset is downloaded automatically via [`kagglehub`](https://github.com/Kaggle/kagglehub) — no manual setup required.

The raw labels are noisy, so the notebooks build a cleaner label set:

- **Drop non-specialty categories** that describe a document type rather than a specialty (e.g. *Surgery*, *SOAP / Chart / Progress Notes*, *Office Notes*, *Discharge Summary*, *Radiology*).
- **Merge overlapping classes** (e.g. *Neurosurgery* → *Neurology*, *Nephrology* → *Urology*), leaving **29 specialties**.

## Pipeline

1. **Cleaning** — lowercase, strip punctuation, remove English stopwords, lemmatize (NLTK, parallelized with multiprocessing).
2. **Class balancing (in the `*_2` / final notebooks)** — the class distribution is heavily skewed (371 Cardiovascular/Pulmonary samples vs. 6 Hospice/Palliative Care), so minority classes are oversampled with lightweight text augmentation: WordNet synonym replacement, random word dropout, and random word swaps.
3. **Vectorization** — depending on the model: TF‑IDF, learned Keras embeddings, or BERT/ClinicalBERT tokenization.
4. **Train/test split** — 80/20, stratified by specialty.
5. **Training & evaluation** — accuracy plus a full per-class precision/recall/F1 report.

## Results

Reported test accuracy from each notebook. Note that the two dataset variants are **not directly comparable**: "balanced" runs apply augmentation before the train/test split, so their test sets contain augmented samples.

### On the balanced (augmented) dataset — 29 specialties

| Notebook | Model(s) | Test accuracy |
|---|---|---|
| `better_data_2.ipynb` | **Logistic Regression (TF‑IDF)** | **91%** |
| `better_data_2.ipynb` | Fine-tuned [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) | 90% |
| `better_data_2.ipynb` | CNN (Conv1D + max pooling) | 89% |
| `rf_biLSTM_GRU_final.ipynb` | Random Forest (TF‑IDF) | 89% |
| `rf_biLSTM_GRU_final.ipynb` | GRU | 89% |
| `lstm_88.ipynb` | LSTM + global max pooling | 88.5% |
| `rf_biLSTM_GRU_final.ipynb` | Bidirectional LSTM | 88% |
| `better_data_2.ipynb` | Multinomial Naive Bayes | 85% |
| `bert-ffnn.ipynb` | Frozen BERT embeddings + feed-forward NN | 84% |

### On the original (unbalanced) dataset

| Notebook | Model(s) | Test accuracy |
|---|---|---|
| `better_data.ipynb` | Fine-tuned ClinicalBERT | 81% |
| `better_data.ipynb` | Logistic Regression (TF‑IDF) | 78% |
| `better_data.ipynb` | Naive Bayes / CNN | 75% / 74% |
| `LSTM2.ipynb` | Bidirectional LSTM (classes with ≥50 samples) | 74% |
| `better_data_LSTM_BERTembed_better.ipynb` | LSTM on BERT embeddings (tuned) | 74% |
| `better_data_biLSTM.ipynb` | Bidirectional LSTM | 61% |
| `better_data_LSTM_BERTembed.ipynb` | LSTM on BERT embeddings | 60% |
| `GRU_better_data.ipynb` | Bidirectional GRU | 48% |

**Takeaways**

- Cleaning the label space (dropping document-type pseudo-classes and merging overlapping specialties) matters more than model choice.
- Balancing classes with simple text augmentation gives every model a large boost.
- A well-tuned TF‑IDF + Logistic Regression baseline is remarkably hard to beat — it matches or outperforms the deep models here at a fraction of the cost.
- Domain-specific pretraining helps: fine-tuned ClinicalBERT is the strongest neural model.

## Getting started

Each notebook is fully self-contained — the first cell installs its dependencies and the dataset is fetched automatically.

```bash
git clone https://github.com/divy-sh/medical-specialty-classification.git
cd medical-specialty-classification
jupyter notebook
```

A good reading order:

1. `better_data.ipynb` — data cleaning + four baseline models on the original data
2. `better_data_2.ipynb` — the same models with augmentation-based class balancing (best results)
3. `rf_biLSTM_GRU_final.ipynb` — Random Forest, Bi-LSTM, and GRU comparison
4. `lstm_88.ipynb`, `bert-ffnn.ipynb`, and the remaining notebooks — further experiments with recurrent models and BERT embeddings

**Main dependencies:** TensorFlow / tf-keras, PyTorch + Hugging Face `transformers` (for BERT notebooks), scikit-learn, NLTK, pandas, kagglehub. A GPU is recommended for the LSTM/GRU and ClinicalBERT notebooks.

