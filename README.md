# Project-NLP-final
 This is our source code for our NLP project
In this repo, there are three folders:
- Finetune T5: All the source code to fine-tune the T5 model
- Encoder - Decoder LSTM: All the source code to train the Encoder - Decoder model
- Errrant_eva: Folder to save file for ERRANT Evaluation

1. Finetune T5:
- Model Test: All the source code to test and evaluate the Fine-Tune T5
+ test_matrice_with_t5.ipnyb: source code to evaluate the T5 with GLEU, BERTScore and ERRANT
+ predict_new.ipnyb: source code to test the Fine-Tune T5 with your input sentence
- Train Model:
+ modeltrain_t5.ipynb: source code to fine-tune the T5 model. 



2. Encoder - Decoder LSTM:
- Model Test: All the source code to test and evaluate the Encoder - Decoder
+ test_matrice_with_lstm.ipynb: source code to evaluate the Encoder - Decoder with GLEU, BERTScore and ERRANT
+ predict_new.py: source code to test the Encoder - Decoder with your input sentence
- Model Building: All the source code to build the Encoder - Decoder:
+ Building LSTM model.ipynb: Full source code to build the LSTM
+ Building LSTM model shorten.ipynb: Shortened source code to build the LSTM

[Note] In order to test the model:
1. T5 Fine-tune:
- Download the model from the link: [T5 Fine-tune]()
- Download the predict_new.ipynb
- Open predict_new, replace the link of the model with your actual model link
- Put your input sentence and try the model
2. Encoder - Decoder LSTM:
- Download the Model Attachement from the link: 
- Download the model from the link: [LSTM Model](https://drive.google.com/file/d/1x93g91Aq8vY3_TcQN_LRBK5AW8AU9qSf/view?usp=sharing)
- Download and open predict_new.py
- Replace the link of the input, output embedding and in, out_tokenizer with your actual path
- Put your input sentence and try the model
