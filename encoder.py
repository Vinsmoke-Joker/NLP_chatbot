import torch.nn as nn
import config
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vocab_size = len(config.input_ws)
        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = config.chatbot_encoder_embedding_dim
        )
        self.gru = nn.GRU(
            input_size = config.chatbot_encoder_embedding_dim,
            hidden_size = config.chatbot_encoder_hidden_size,
            num_layers = config.chatbot_encoder_numlayers,
            bidirectional = config.chatbot_encoder_bidirectional,
            dropout = config.chatbot_encoder_dropout,
            batch_first = True
        )


    def forward(self, input, input_length):
        # 经过embedding input[batch_size, seq_len] ->[batch_size, seq_len, embedding_dim]
        input_embeded = self.embedding(input)
        # 进行打包 input_packed [batch_size, seq_len, embedding_dim]
        input_packed = nn.utils.rnn.pack_padded_sequence(input = input_embeded, lengths = input_length, batch_first = True)
        # 通过GRU  output ->[batch_size, seq_len, hidden_size * 2]
        # hidden ->[numlayer * 2, batch_size, hidden_size]
        output, hidden = self.gru(input_packed)
        # 进行解包 encoder_output ->[batch_size, seq_len, hidden_size]
        encoder_output, output_length = nn.utils.rnn.pad_packed_sequence(sequence = output, batch_first = True, padding_value = config.input_ws.PAD)
        # 由于双向GRU 对hidden进行拼接
        # hidden[num_layer * 2, batch_size, hidden_size]
        # hidden[-1] == hidden[-1, :, :]
        # 经过torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim = -1)
        # encoder_hidden = [batch_size, hidden_size * 2]
        # 由于decoder 输入要求是三维，对encoder_hidden扩展维度为[1, batch_size, hidden_size * 2]
        encoder_hidden = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim = -1).unsqueeze(0)
        return encoder_output, encoder_hidden