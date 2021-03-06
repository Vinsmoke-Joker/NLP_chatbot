import torch
import torch.nn as nn
import config
from attention import Attention
import random
import torch.nn.functional as F
import numpy as np


class Decoder_Attn(nn.Module):
    def __init__(self):
        super(Decoder_Attn, self).__init__()
        self.vocab_size = len(config.target_ws)
        # encoder中为双向GRU,hidden进行了双向拼接,为了attention计算方便
        # hidden_size = en_hidden_size
        self.hidden_size = config.chatbot_encoder_hidden_size
        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = config.chatbot_decoder_embedding_dim
        )
        self.gru = nn.GRU(
            input_size = config.chatbot_decoder_embedding_dim,
            hidden_size = self.hidden_size * 2,
            num_layers = config.chatbot_decoder_numlayers,
            batch_first = True,
            bidirectional = False
        )
        # 处理forward_step中decoder的每个时间步输出形状
        self.fc = nn.Linear(self.hidden_size * 2, self.vocab_size)
        # 实例化attn_weights
        self.attn = Attention(method = 'general')
        # self.attn 形状为
        self.fc_attn = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)


    def forward(self, encoder_hidden, target, encoder_outputs):
        """
        :param encoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param target: [batch_size, en_seq_len]
        添加attention的decoder中, 对encoder_outputs进行利用与decoder_hidden计算得到注意力表示
        encoder_outputs为新增参数
        :param encoder_outputs: [batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        batch_size = encoder_hidden.size(1)
        # 初始化一个[batch_size, 1]的SOS作为decoder第一个时间步的输入decoder_input
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)
        # 初始化一个[batch_size, de_seq_len, vocab_size]的张量拼接每个time step结果
        decoder_outputs = torch.zeros([batch_size, config.chatbot_target_max_len, self.vocab_size]).to(config.device)
        # encoder_hidden 作为decoder的第一个时间步的hidden
        decoder_hidden = encoder_hidden
        # 按照teacher_forcing的更新策略进行每个时间步的更新
        teacher_forcing = random.random() > 0.5
        if teacher_forcing:
            # 对每个时间步进行遍历
            for t in range(config.chatbot_target_max_len):
                # decoder_output_t [batch_size, vocab_size]
                # decoder_hidden [1, batch_size, en_hidden_size * 2]
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                # 进行拼接每个时间步
                decoder_outputs[:, t, :] = decoder_output_t
                # 使用teacher_forcing,下一次采用真实结果
                # target[:, t] [batch_size]
                # decoder_input [batch_size, 1]
                decoder_input = target[:, t].unsqueeze(1)
        else:
            for t in range(config.chatbot_target_max_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, t, :] = decoder_output_t
                # 不使用teacher_forcing下次采用预测结果
                index = torch.argmax(decoder_output_t, dim = -1)
                # index [batch_size]
                # decoder_input [batch_size, 1]
                decoder_input = index.unsqueeze(-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        每个时间步的处理
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param encoder_outputs [batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        # 依次通过embedding、gru 和fc 最终返回log_softmax
        # decoder_input_embeded [batch_size, 1] -> [batch_size, 1, embedding_dim]
        decoder_input_embeded = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.gru(decoder_input_embeded, decoder_hidden)
        """通过decoder_hidden和encoder_outputs进行attention计算"""
        # 1.通过attention 计算attention weight
        # decoder_hidden [1, batch_size, en_hidden_size * 2]
        # encoder_outputs [batch_size, en_seq_len, en_hidden_size * 2]
        # attn_weight [batch_size, en_seq_len]
        attn_weight = self.attn(decoder_hidden, encoder_outputs)
        # 2.attn_weight与encoder_outputs 计算得到上下文向量
        # encoder_outputs[batch_size, en_seq_len, en_hidden_size * 2]
        # attn_weight [batch_size, en_seq_len]
        # 二者进行矩阵乘法,需要对attn_weight进行维度扩展->[batch_size, 1, en_seq_len]
        # context_vector [batch_size, 1, en_hidden_size * 2]
        context_vector = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)
        # 3.context_vector 与decoder每个时间步输出decoder_output进行拼接和线性变换，得到每个时间步的注意力结果输出
        # decoder_output[batch_size, 1, en_hidden_size * 2]
        # context_vector[batch_size, 1, en_hidden_size * 2]
        # 拼接后形状为 [batch_size, 1, en_hidden_size * 4]
        # 由于要进行全连接操作,需要对拼接后形状进行降维unsqueeze(1)
        # ->[batch_size, en_hidden_size * 4]
        # 且decoder每个时间步输出结果经过self.fc的维度为[batch_size, vocab_size]
        # 因此，self.attn的fc 输入输出维度为(en_hidden_size * 4, en_hidden_size * 2)
        # self.fc输入输出维度为(en_hidden_size * 2, vocab_size)
        # 注意:这里用torch.tanh,使用F.tanh会 'nn.functional.tanh is deprecated. Use torch.tanh instead'
        attn_result = torch.tanh(self.fc_attn(torch.cat((decoder_output, context_vector),dim = -1).squeeze(1)))
        # attn_result [batch_size, en_hidden_size * 2]
        # 经过self.fc后改变维度
        # decoder_output_t [batch_size, en_hidden_size * 2]->[batch_size, vocab_size]
        decoder_output_t = F.log_softmax(self.fc(attn_result), dim = -1)
        # decoder_hiddem [1, batch_size, en_hidden_size * 2]
        return decoder_output_t, decoder_hidden


    def evaluate(self, encoder_hidden, encoder_outputs):
        """
        评估逻辑
        :param encoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param encoder_outputs:  [batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        batch_size = encoder_hidden.size(1)
        # 初始化decoder第一个时间步的输入和hidden和decoder输出
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)
        decoder_outputs = torch.zeros((batch_size, config.chatbot_target_max_len, len(config.target_ws))).to(config.device)
        decoder_hidden = encoder_hidden
        # 初始化用于存储的预测序列
        predict_result = []
        for t in range(config.chatbot_target_max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[:, t, :] = decoder_output_t
            index = torch.argmax(decoder_output_t, dim = -1)
            decoder_input = index.unsqueeze(-1)
            predict_result.append(index.cpu().detach().numpy())
        predict_result = np.array(predict_result).transpose()
        return decoder_outputs, predict_result