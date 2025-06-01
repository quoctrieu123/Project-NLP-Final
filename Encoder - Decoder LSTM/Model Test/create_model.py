import torch
import numpy as np #thư viện dùng để xử lý mảng nhiều chiều
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length, link_to_in_embedding):
        super(Encoder, self).__init__()
        self.vocab_size = inp_vocab_size #số lượng từ vựng
        self.embedding_size = embedding_size
        self.lstm_units = lstm_size
        self.input_length = input_length
        in_embedding_matrix = np.load(link_to_in_embedding)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(in_embedding_matrix), freeze=True,padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_size, self.lstm_units,batch_first=True)
    def forward(self,input_sequence,states):
        input_embedding = self.embedding(input_sequence)
        self.lstm_output, (self.state_h, self.state_c) = self.lstm(input_embedding, states)
        return self.lstm_output, self.state_h, self.state_c
    def initialize_states(self,batch_size):
        lstm_state_h = torch.zeros((1,batch_size,self.lstm_units)).to(device)
        lstm_state_c = torch.zeros((1,batch_size,self.lstm_units)).to(device)
        return lstm_state_h, lstm_state_c
    
def dot_func(encoder_output,decoder_hidden_state):
    decoder_hidden_state = torch.reshape(decoder_hidden_state,[decoder_hidden_state.shape[1],1,decoder_hidden_state.shape[2]])
    dot_product = torch.matmul(encoder_output,decoder_hidden_state.transpose(1,2))
    return dot_product

class Attention(nn.Module):
    def __init__(self,scoring_function, att_units):
        super(Attention,self).__init__()
        self.scoring_function = scoring_function
        self.att_units = att_units
        self.timesteps = 0
    def forward(self,decoder_hidden_state,encoder_output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_output = encoder_output.to(device)
        decoder_hidden_state = decoder_hidden_state.to(device)
        if self.scoring_function == 'dot':
            alpha = F.softmax(dot_func(encoder_output,decoder_hidden_state),dim = 1)
            c_t = torch.sum(alpha *encoder_output,dim = 1)
            return c_t,alpha
        
class One_Step_Decoder(nn.Module):
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units, link_to_out_embedding):
        super(One_Step_Decoder,self).__init__()

        self.tar_vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        out_embedding_matrix = np.load(link_to_out_embedding)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(out_embedding_matrix), freeze=True,padding_idx=0)
        self.lstm = nn.LSTM(self.dec_units + self.embedding_dim,self.dec_units,batch_first=True)
        self.dense = nn.Linear(self.dec_units,self.tar_vocab_size)
    def forward(self,input_to_decoder,encoder_output,state_h,state_c):
        input_embedding = self.embedding(input_to_decoder)
        if self.score_fun == "dot":
            attention = Attention("dot",self.att_units)
            context_vector,attention_weights = attention(state_h,encoder_output)
        out = torch.cat([input_embedding,context_vector.unsqueeze(1)],dim = 2)
        self.lstm_output, (self.state_h, self.state_c_) = self.lstm(out, (state_h,state_c))
        result_out = self.dense(self.lstm_output)
        return result_out.squeeze(1), self.state_h, self.state_c_, attention_weights, context_vector
    
class Decoder(nn.Module):
    def __init__(self,out_vocab_size,embedding_dim,input_length,dec_units, score_fun,att_units, link_to_out_embedding):
        super(Decoder,self).__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.one_step_decoder = One_Step_Decoder(self.out_vocab_size, self.embedding_dim, self.input_length, self.dec_units ,self.score_fun ,self.att_units, link_to_out_embedding= link_to_out_embedding)
    def forward(self,input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_to_decoder = input_to_decoder.to(device)
        encoder_output = encoder_output.to(device)
        decoder_hidden_state = decoder_hidden_state.to(device)
        decoder_cell_state = decoder_cell_state.to(device)

        batch_size, timesteps = input_to_decoder.shape
        out_array = torch.zeros((batch_size, timesteps, self.one_step_decoder.tar_vocab_size)).to(device)

        for timestep in range(timesteps):
            output,decoder_hidden_state,decoder_cell_state,_,_ = self.one_step_decoder(input_to_decoder[:,timestep:timestep+1],encoder_output,decoder_hidden_state,decoder_cell_state)
            out_array[:,timestep,:] = output
        return out_array
    
INPUT_VOCAB_SIZE = 77159
OUTPUT_VOCAB_SIZE = 60992
INPUT_ENCODER_LENGTH = 34
INPUT_DECODER_LENGTH = 42
class encoder_decoder(nn.Module):
    def __init__(self,enc_units,dec_units,scoring_func,att_units,link_to_in_embedding,link_to_out_embedding):
        super(encoder_decoder,self).__init__()
        self.scoring_func = scoring_func
        self.att_units = att_units
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.link_to_in_embedding = link_to_in_embedding
        self.link_to_out_embedding = link_to_out_embedding
        self.encoder = Encoder(INPUT_VOCAB_SIZE, embedding_size = 300, lstm_size= self.enc_units , input_length= INPUT_ENCODER_LENGTH, link_to_in_embedding = self.link_to_in_embedding)
        self.decoder = Decoder(OUTPUT_VOCAB_SIZE, embedding_dim=300, input_length = None, dec_units= self.dec_units ,score_fun =self.scoring_func,att_units = self.att_units, link_to_out_embedding = link_to_out_embedding)
    
    def forward(self,data):
        input,output = data[0],data[1]
        states = self.encoder.initialize_states(input.shape[0])
        encoder_output,encoder_final_state_h,encoder_final_state_c = self.encoder(input, states)
        decoder_output = self.decoder(output,encoder_output, encoder_final_state_h, encoder_final_state_c)
        return decoder_output

def create_model(link_to_in_embedding,link_to_out_embedding):
    model = encoder_decoder(enc_units=512,dec_units=512,scoring_func="dot",att_units=256,link_to_in_embedding= link_to_in_embedding,link_to_out_embedding= link_to_out_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model
