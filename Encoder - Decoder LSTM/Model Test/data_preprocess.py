import re
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch
import pickle
import unicodedata

INPUT_VOCAB_SIZE = 77159
OUTPUT_VOCAB_SIZE = 60992
INPUT_ENCODER_LENGTH = 34
INPUT_DECODER_LENGTH = 42
def input_processor(input_sentence,tokenizer, pad_seq): #chuyển câu đầu vào và chuyển thành index, thêm padding cần thiết để đưa vào encoder (phục vụ cho test model)

  encoder_input = preprocess(input_sentence, add_start_token= True, add_end_token=True)

  tokenized_text = tokenizer.texts_to_sequences([encoder_input])
  if pad_seq == True:
    tokenized_text = pad_sequences(tokenized_text, maxlen=INPUT_ENCODER_LENGTH, padding="post")

  tokenized_text = tf.convert_to_tensor(tokenized_text, dtype = tf.float32)
  return tokenized_text

def preprocess(t, add_start_token, add_end_token):
  t = str(t)
  if add_start_token == True and add_end_token == False: #nếu thêm start token và không thêm end token
    t = '<start>'+' '+t
  if add_start_token == False and add_end_token == True: #nếu không thêm start token và thêm end token
    t = t+' '+'<end>'
  if add_start_token == True and add_end_token == True: #nếu thêm cả start token và end token
    t = '<start>'+' '+t+' '+'<end>'

  t = re.sub(' +', ' ', t) #loại bỏ khoảng trắng thừa
  return t


def predict_sentence(model,input_sentence, link_to_in_tokenizer,link_to_out_tokenizer, max_length=34, device= "cuda" if torch.cuda.is_available() else "cpu"):
    # Load tokenizer từ file
    input_sentence = str(input_sentence)  # Đảm bảo input là chuỗi
    input_sentence = clean_text(input_sentence)  # Làm sạch câu đầu vào
    tokenizer = load_tokenizer(link_to_in_tokenizer)
    sequence = tokenizer.texts_to_sequences([input_sentence])[0]
    if len(sequence) > 34: #nếu câu đầu vào dài hơn 30 thì không sửa
        return input_sentence
    out_tokenizer = load_tokenizer(link_to_out_tokenizer)
    model.eval()  # Chuyển model sang chế độ eval

    start_token_in = 1  # Start token của tokenizer (input)
    end_token_in = 2    # End token của tokenizer (input)
    start_token_out = 3  # Start token của out_tokenizer (output)
    end_token_out = 4    # End token của out_tokenizer (output)

    # 1️⃣ Xử lý đầu vào (encoder_input)
    input_tensor = input_processor(input_sentence, tokenizer, pad_seq=True)  # Đưa vào hàm tiền xử lý có sẵn
    input_tensor = torch.tensor(input_tensor.numpy(), dtype=torch.long).to(device)  # (1, 17)

    # 2️⃣ Khởi tạo decoder_input với start token
    decoder_input = torch.tensor([[start_token_out]], dtype=torch.long).to(device)  # (1,1)

    predicted_sentence = []

    with torch.no_grad():  # Không cần tính gradient khi inference
        for _ in range(max_length):  # Giới hạn tối đa 29 từ
            # 3️⃣ Dự đoán từ tiếp theo
            output = model([input_tensor, decoder_input])  # (1, seq_len, vocab_size)

            # 4️⃣ Chọn từ có xác suất cao nhất (Greedy Search)
            next_word = torch.argmax(output[:, -1, :], dim=-1).item()

            # 5️⃣ Nếu gặp token <end> thì dừng lại
            if next_word == end_token_out:
                break

            # 6️⃣ Thêm từ vào kết quả
            predicted_sentence.append(next_word)

            # 7️⃣ Cập nhật decoder_input để tiếp tục sinh từ mới
            next_word_tensor = torch.tensor([[next_word]], dtype=torch.long).to(device)  # (1,1)
            decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)  # Nối thêm từ vào decoder_input

            # 8️⃣ Nếu đã đạt độ dài tối đa 29 thì dừng
            if decoder_input.shape[1] >= max_length:
                break

    # 9️⃣ Chuyển index thành câu hoàn chỉnh sử dụng out_tokenizer
    predicted_words = [out_tokenizer.index_word.get(idx, "<unk>") for idx in predicted_sentence]
    
    return " ".join(predicted_words)

def load_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    


def decontracted(phrase): #hàm chuyển từ viết tắt thành từ đầy đủ
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"gon na", " going to", phrase)
    phrase = re.sub(r"wan na", " want to", phrase)
    phrase = re.sub(r"gonna", " going to", phrase)
    phrase = re.sub(r"wanna", " want to", phrase)


    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub("\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

def clean_text(t): #hàm bỏ dấu và ký tự unicode

  t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii') 
  t = decontracted(t)

  t = re.sub(r'x D', '', t)
  t = re.sub(r': D', '', t)
  t = re.sub(r': P', '', t)

  t = re.sub(r'xD', '', t)
  t = re.sub(r':D', '', t)
  t = re.sub(r':P', '', t)

  if '(' in t and ')' in t: #loại bỏ nội dung trong dấu ngoặc đơn => Hello (world) => Hello
    try:
      t = re.sub(t.split("(")[-1].split(")")[0], '', t)
    except:
      pass
    #t = re.sub("(", '', t)
    #t = re.sub(")", '', t)

  t = re.sub(r'[^A-Za-z;!?.,\-\' ]+', ' ', t) #loại bỏ các emoji, số, ký tự đặc biệt

#chuẩn hóa dấu câu, thêm khoảng trống trước dâu câu. 
  t = re.sub(r'\.+',r' .',t) #tôi yêu viêt nam. => tôi yêu viêt nam .
  t = re.sub(r'\;+',r' , ',t)
  t = re.sub(r'!+',r' !',t )
  t = re.sub(r'\?+',r' ?',t )
  t = re.sub(r'\-+',r' - ',t )
  t = re.sub(r'\,+',r' , ',t )
  t = re.sub(r'\'+',r" ' ",t)
  t = re.sub(' +', ' ', t) #loại bỏ khoảng trắng thừa vd: "hello   world" => "hello world"

  return t #trả về văn bản được làm sạch