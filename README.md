
---

## Our Datasets

- **Training Dataset – C4 (Common Crawl)**  
  [Download on Kaggle](https://www.kaggle.com/datasets/dariocioni/c4200m)

- **Evaluation Dataset – Lang-8**  
  [Download on Kaggle](https://www.kaggle.com/datasets/studentramya/lang-8)

---

## 1. Fine-tuned T5

### Contents:
- `modeltrain_t5.ipynb`: Fine-tunes the T5 model on the C4 dataset.
- `test_matrice_with_t5.ipynb`: Evaluates the T5 model using GLEU, BERTScore, and ERRANT.
- `predict_new.ipynb`: Run inference on new input sentences using your fine-tuned T5 model.

### How to Run:
1. Download the fine-tuned model:  
   [T5 Fine-tune](https://drive.google.com/drive/folders/16ojRM38ZUNO40iIKytgATGPuk8aJDhBe?usp=sharing)
2. Open `predict_new.ipynb`
3. Replace the model path with your local directory
4. Enter any input sentence to see the correction results

---

## 2. Encoder–Decoder LSTM

### Contents:
- `Building LSTM model.ipynb`: Full training code for the LSTM-based GEC model.
- `Building LSTM model shorten.ipynb`: Simplified version for quick implementation.
- `test_matrice_with_lstm.ipynb`: Evaluation file (GLEU, BERTScore, ERRANT).
- `predict_new.py`: Script to perform inference using a trained LSTM model.

### Pre-trained Embeddings:
- While training, we used FastText pre-trained word vectors downloaded from [Facebook AI](https://fasttext.cc/docs/en/crawl-vectors.html)

### Preprocessed Data:
[Download Preprocessed Data](https://drive.google.com/drive/folders/1EFWKW6SiPnbPmsjHoHP4qpdQcDe6kJll?usp=sharing)
- `train_dataset.pt`: final preprocessed dataset for training
- `test_dataset.pt`: final preprocessed dataset for testing
- `cleaned_data.csv`: final preprocessed dataset for evaluation

### Model Attachments:
[Download Model Attachments](https://drive.google.com/drive/folders/16G99qkbqIItvv0RmBNfcjlb-mQ73G-mF?usp=sharing)
- `in_tokenizer`, `out_tokenizer`: file saved the model's tokenizer
- `in_embedding_matrix`, `out_embedding_matrix`: file saved the model's embedding matrix

### Trained Model:
Download Trained LSTM Model](https://drive.google.com/file/d/1x93g91Aq8vY3_TcQN_LRBK5AW8AU9qSf/view?usp=sharing)

### How to Run:
1. Download the model and all attachments.
2. Open `predict_new.py`
3. Update paths for embeddings and tokenizers
4. Input a sentence to see the predicted correction

---

## 3. Evaluation Tools

- Evaluation is performed using:
  - **GLEU**
  - **BERTScore**
  - **ERRANT**

Refer to the respective `test_matrice_with_*.ipynb` notebooks in each model folder.
