import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Attention(nn.Module):
    def __init__(self, method):
        """
        attention 机制
        :param method:三种attention_weights 计算方法general, dot, concat
        """
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = config.chatbot_encoder_hidden_size
        assert self.method in ['dot', 'general', 'concat'], 'attention method error'
        if self.method == 'dot':
            # dot 为decoder_hidden 和encoder_outputs 直接进行矩阵乘法
            pass
        elif self.method == 'general':
            # general为对decoder_hidden 进行矩阵变换后，与encoder_outputs相乘
            self.Wa = nn.Linear(config.chatbot_encoder_hidden_size * 2, config.chatbot_encoder_hidden_size * 2,
                                bias=False)
        elif self.method == 'concat':
            self.Wa = nn.Linear(config.chatbot_encoder_hidden_size * 4, config.chatbot_encoder_hidden_size * 2,
                                bias=False)
            self.Va = nn.Linear(config.chatbot_encoder_hidden_size * 2, 1, bias = False)


    def forward(self, decoder_hidden, encoder_outputs):
        """
        进行三种运算得到attn_weights
        :param decoder_hidden: decoder每个时间步的隐藏状态[1, batch_size, en_hidden_size * 2]
        由于encoder中使用Bi-GRU,最后对双向hidden进行了拼接,因此de_hidden_size = en_hidden_size * 2
        未拼接前 encoder_hidden [1, batch_size, en_hidden_size]
        :param encoder_outputs:encoder最后的输出[batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        if self.method == 'dot':
            return self.dot_score(decoder_hidden, encoder_outputs)
        elif self.method == 'general':
            return self.general_score(decoder_hidden, encoder_outputs)
        elif self.method == 'concat':
            return self.concat_score(decoder_hidden, encoder_outputs)


    def dot_score(self, decoder_hidden, encoder_outputs):
        """
        dot 方法：直接对decoder_hidden 和 encoder_outputs进行矩阵乘法
        :param decoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param encoder_outputs:[batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        # 要进行矩阵乘法,需要改变decoder_hidden的形状为[batch_size, en_hidde_size * 2 , 1]
        # 乘法后形状为[batch_size, en_seq_len, 1]
        # squeeze去掉1的维度 为[batch_size, en_seq_len]
        # 最终对结果在en_seq_len维度上进行log_softmax
        return F.log_softmax(torch.bmm(encoder_outputs, decoder_hidden.permute(1, 2, 0)).squeeze(-1), dim = -1)


    def general_score(self, decoder_hidden, encoder_outputs):
        """
        general 方法：对decoder_hidden进行线性变换后与encoder_outputs进行矩阵乘法
        :param decoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param encoder_outputs: [batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        # 由于要进行线性变换, decoder_hidden首先变成二维张量,因此线性变换的输入维度为en_hidden_size * 2
        # [1, batch_size, en_hidden_size * 2]->[batch_size, en_hidden_size * 2]
        decoder_hidden = decoder_hidden.squeeze(0)
        # 由于要与encoder_outputs进行矩阵计算,需要将decoder_hidden的形状改变为dot中的形状
        # 即[batch_size, en_hidden_size * 2, 1],因此线性变换的输出维度为en_hidden_size * 2
        decoder_hidden = self.Wa(decoder_hidden).unsqueeze(-1)
        # 进行矩阵乘法[batch_size, en_seq_len, 1] ->squeeze [batch_size, en_seq_len]
        # torch.bmm 注意矩阵形状, 参数位置需要 根据矩阵乘法要求,不能写反
        return F.log_softmax(torch.bmm(encoder_outputs, decoder_hidden).squeeze(-1), dim = -1)


    def concat_score(self, decoder_hidden, encoder_outputs):
        """
        concat方法：decoder_hidden和encoder_outputs拼接，
        把这个结果使用tanh进行处理后的结果进行对齐(进行矩阵乘法，变换为需要的形状)计算之后，
        和encoder outputs进行矩阵乘法
        :param decoder_hidden: [1, batch_size, en_hidden_size * 2]
        :param encoder_outputs: [batch_size, en_seq_len, en_hidden_size * 2]
        :return:
        """
        encoder_seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        # repeat 沿着该维度重复指定次数
        # repeat(3,1,1)指在0维度重复3次,其他2个维度各一次
        # decoder_hidden [1, batch_size, en_hidden_size *2]->squeeze(0):[batch_size, en_hidden_size * 2]
        # ->repeat:[encoder_seq_len, batch_size, en_hidden_size * 2] ->transpose:[batch_size, encoder_seq_len, en_hidden_size * 2]
        decoder_hidden_repeated = decoder_hidden.squeeze(0).repeat(encoder_seq_len, 1, 1).transpose(1,0)
        # 对decoder_hidden_repeated和encoder_outputs进行拼接
        # cat:[batch_size, en_seq_len, en_hidden_size * 2 *2]
        # view[batch_size * en_seq_len, en_hidden_size * 4]
        # 因此第一个线性层输入维度为en_hidden_size * 4
        h_cated = torch.cat((decoder_hidden_repeated, encoder_outputs), dim = -1).view(batch_size * encoder_seq_len, -1)
        # 拼接后，需要进行线性变换及tanh和第二次线性变换最终将结果变为[batch_size, en_seq_len]
        # h_cated->Wa:[batch_size * en_seq_len, en_hidden_size *4] ->[batch_size * en_seq_len, en_hidden_size *2]
        # ->Va:[batch_size * en_seq_len, en_hidden_size *2] ->[batch_size * en_seq_len, 1]
        # ->view:[batch_size * en_seq_len, 1] ->[batch_size ,en_seq_len]
        attn_weight = self.Va(torch.tanh(self.Wa(h_cated))).view([batch_size, encoder_seq_len])
        return F.log_softmax(attn_weight, dim = -1)
