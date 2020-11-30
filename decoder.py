import torch.nn as nn
import torch
import config
import torch.nn.functional as F
import random
import numpy as np


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.vocab_size = len(config.target_ws)
        self.hidden_size = config.chatbot_encoder_hidden_size * 2
        self.embedding = nn.Embedding(
            num_embeddings = len(config.target_ws),
            embedding_dim = config.chatbot_decoder_embedding_dim
        )
        self.gru = nn.GRU(
            input_size = config.chatbot_decoder_embedding_dim,
            # encoder中采用双向GRU, 在最后进行了双向拼接，decoder中hidden为encoder_hidden * 2
            # 以下注释中hidden_size, 均为decoder中hidden_size
            hidden_size = self.hidden_size,
            num_layers = config.chatbot_decoder_numlayers,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(self.hidden_size ,self.vocab_size)


    def forward(self, encoder_hidden, target):
        """
        :param encoder_hidden:  [1, batch_size, hidden_size]
        :param target: [batch_size, seq_len]
        :return:
        """
        batch_size = encoder_hidden.size(1)
        # 初始化一个[batch_size, 1]的全SOS张量,作为decoder的第一个time step输入
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)
        # encoder_hidden[1, batch_size, hidden_size] 作为decoder 第一个time step 的输入
        decoder_hidden = encoder_hidden
        # 初始化一个[batch_size, seq_len, vocab_size]的outputs 存储每个时间步结果
        decoder_outputs = torch.zeros([batch_size, config.chatbot_target_max_len, self.vocab_size]).to(config.device)
        # 判断是否使用teacher_forcing
        if random.random() > config.teacher_forcing:
            # 进行每个时间步的遍历
            for t in range(config.chatbot_target_max_len):
                output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                # 获取每个时间步t的结果
                decoder_outputs[:, t, :] = output_t
                # 若不使用teacher_forcing 选取每个时间步的预测值最为下个时间步输入
                # index[batch_size, 1]
                """
                若使用max()
                value, index = output_t.max(dim = -1) # [batch_size]
                decoder_input = index.unsqueeze(1)
                若是argmax()
                index = output_t.argmax(dim = -1) # [batch_size]
                decoder_input = index.unsqueeze(1)
                """
                value, index = torch.topk(output_t, k = 1)
                # 需要保证decoder_input的输入为[batch_size, 1]
                decoder_input = index
        else:
            for t in range(config.chatbot_target_max_len):
                output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = output_t
                # 若使用teacher_forcing,采用真实值作为下次输入
                # 使得decoder_input的形状为[batch_size, 1]
                decoder_input = target[:,t].unsqueeze(1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, decoder_input, decoder_hidden):
        """
        处理每个时间步逻辑
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, hidden_size]
        :return:
        """
        # [batch_size, 1] ->[batch_size, 1, embedding_dim]
        decoder_input_embeded = self.embedding(decoder_input)
        # decoder_output_t [batch_size, 1, embedding_dim] ->[batch_size, 1, hidden_size]
        # decoder_hidden_t [1,batch_size, hidden_size]
        decoder_output_t, decoder_hidden_t = self.gru(decoder_input_embeded, decoder_hidden)
        # 对decoder_output_t进行fcq前，需要对其进行形状改变 [batch_size, hidden_size]
        decoder_output_t = decoder_output_t.squeeze(1)
        # 进行fc -> [batch_size, vocab_size]
        decoder_output_t = F.log_softmax(self.fc(decoder_output_t), dim = -1)
        return decoder_output_t, decoder_hidden_t


    def evaluate(self, encoder_hidden):
        """
        评估, 和fowward逻辑类似
        :param encoder_hidden: encoder最后time step的隐藏状态 [1, batch_size, hidden_size]
        :return:
        """
        batch_size = encoder_hidden.size(1)
        # 初始化一个[batch_size, 1]的SOS张量,作为第一个time step的输出
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)
        # encoder_hidden 作为decoder第一个时间步的hidden [1, batch_size, hidden_size]
        decoder_hidden = encoder_hidden
        # 初始化[batch_size, seq_len, vocab_size]的outputs 拼接每个time step结果
        decoder_outputs = torch.zeros((batch_size, config.chatbot_target_max_len, self.vocab_size)).to(config.device)
        # 初始化一个空列表,存储每次的预测序列
        predict_result = []
        # 对每个时间步进行更新
        for t in range(config.chatbot_target_max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # 拼接每个time step,decoder_output_t [batch_size, vocab_size]
            decoder_outputs[:, t, :] = decoder_output_t
            # 由于是评估,需要每次都获取预测值
            index = torch.argmax(decoder_output_t, dim = -1)
            # 更新下一时间步的输入
            decoder_input = index.unsqueeze(1)
            # 存储每个时间步的预测序列
            predict_result.append(index.cpu().detach().numpy()) # [[batch], [batch]...] ->[seq_len, vocab_size]
        # 结果转换为ndarry,每行是一个预测结果即单个字对应的索引, 所有行为seq_len长度
        predict_result = np.array(predict_result).transpose()  # (batch_size, seq_len)的array
        return decoder_outputs, predict_result
