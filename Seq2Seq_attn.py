from encoder import Encoder
from decoder_attn import Decoder_Attn
import torch.nn as nn


class Seq2Seq_Attn(nn.Module):
    def __init__(self):
        super(Seq2Seq_Attn, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder_Attn()


    def forward(self, input, target, input_length, target_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        decoder_output, decoder_hidden = self.decoder(encoder_hidden, target, encoder_output)
        return decoder_output


    def evaluation_attn(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        decoder_output, predict_result = self.decoder.evaluate(encoder_hidden, encoder_output)
        return decoder_output, predict_result