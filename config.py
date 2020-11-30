import pickle 
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""word2sequence"""

chatbot_train_batch_size = 200
chatbot_test_batch_size = 300

input_ws = pickle.load(open('./model/ws_input.pkl', 'rb'))
target_ws = pickle.load(open('./model/ws_target.pkl' ,'rb'))

chatbot_input_max_len = 20
chatbot_target_max_len = 30


"""Encoder"""
chatbot_encoder_embedding_dim = 300
chatbot_encoder_hidden_size = 128
chatbot_encoder_numlayers = 2
chatbot_encoder_bidirectional = True
# RNN中 若要添加dropout, num_layer >= 2
chatbot_encoder_dropout = 0.3

"""Decoder"""
chatbot_decoder_embedding_dim = 300
chatbot_decoder_numlayers = 1
teacher_forcing = 0.5

"""beam search"""
beam_width = 2