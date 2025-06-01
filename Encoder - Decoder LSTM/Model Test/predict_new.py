from create_model import create_model
import torch
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from data_preprocess import input_processor, preprocess, predict_sentence
link_to_in_tokenizer = r"C:\Users\Admin\Downloads\Model_NLP\NLP\Model Attachment\tokenizer.pickle"
link_to_out_tokenizer = r"C:\Users\Admin\Downloads\Model_NLP\NLP\Model Attachment\out_tokenizer.pickle"
link_to_in_embedding = r"C:\Users\Admin\Downloads\Model_NLP\NLP\Model Attachment\in_embedding.npy" #chủ động thay đổi đường dẫn tới file
link_to_out_embedding = r"C:\Users\Admin\Downloads\Model_NLP\NLP\Model Attachment\out_embedding.npy"#chủ động thay đổi đường dẫn tới file
model = create_model(link_to_in_embedding,link_to_out_embedding) #load model đã train
model.load_state_dict(torch.load(r"C:\Users\Admin\Downloads\Model_NLP\NLP\Model Saved\best_model.pth")) #chủ động thay đổi đường dẫn tới file
input_sentence = "i want eat banana." #câu cần sửa
output_sentence = predict_sentence(model,input_sentence, link_to_in_tokenizer, link_to_out_tokenizer) #dự đoán câu
print("Input Sentence:", input_sentence)
print("Output Sentence:", output_sentence)

output = []
test_sentences = []

r"""
with open(r"C:\Users\Admin\Downloads\Grammar Correction\m2scorer\m2scorer\example\system_input.txt", "r", encoding="utf-8") as f:
    for sentence in f:
        test_sentences.append(sentence)

for sentence in test_sentences:
    sentence = sentence.strip()
    output_sentence = predict_sentence(model, sentence, link_to_in_tokenizer, link_to_out_tokenizer) #dự đoán câu
    output.append(output_sentence)

with open(r"C:\Users\Admin\Downloads\Grammar Correction\m2scorer\m2scorer\example\system_output.txt", "w", encoding="utf-8") as f:
    for sentence in output:
        sentence = sentence.capitalize()
        f.write(sentence + "\n")



import pandas as pd
data = pd.read_csv(r"C:\Users\Admin\Downloads\Grammar Correction\C4_200M.tsv-00001-of-00010", sep='\t', header=None)
data = data[:300]
data = data[data['error'].str.split().str.len() <= 32]
data.rename(columns={0: "error", 1: "correct"}, inplace=True)
error_sentences = [str(sentence) for sentence in data["error"]]
ref_sentences = [str(sentence) for sentence in data["correct"]]
import torch

outputs_sentences = []
for sentence in error_sentences[:300]:
    decoded = predict_sentence(model, sentence, link_to_in_tokenizer, link_to_out_tokenizer)
    outputs_sentences.append(decoded)

ref_sentences = [sentence.lower() for sentence in ref_sentences]
error_sentences = [sentence.lower() for sentence in error_sentences]

import nltk
from nltk.util import ngrams
from collections import Counter

# Nếu chưa cài đặt bộ tokenizer:
# nltk.download('punkt')

def sentence_gleu(source: str, hypothesis: str, reference: str, max_n: int = 4) -> float:
    source_tokens = nltk.word_tokenize(source)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    ref_tokens = nltk.word_tokenize(reference)
    
    overlap = 0
    total = 0

    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(ngrams(hyp_tokens, n))
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        src_ngrams = Counter(ngrams(source_tokens, n))

        ref_diff = ref_ngrams - src_ngrams
        hyp_diff = hyp_ngrams - src_ngrams

        match = sum((hyp_diff & ref_diff).values())
        total_hyp = sum(hyp_diff.values())

        overlap += match
        total += total_hyp

    if total == 0:
        return 1.0 if sum((ref_ngrams - src_ngrams).values()) == 0 else 0.0
    return overlap / total

def compute_corpus_gleu(sources, predictions, references, max_n=4):

    assert len(sources) == len(predictions) == len(references), "Danh sách không khớp độ dài"

    scores = [
        sentence_gleu(src, pred, ref, max_n=max_n)
        for src, pred, ref in zip(sources, predictions, references)
    ]
    return sum(scores) / len(scores)

print(compute_corpus_gleu(error_sentences[:300], outputs_sentences[:], ref_sentences[:300]))
"""