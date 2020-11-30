from encoder import Encoder
from decoder_beam import Decoder_Attn_Beam
import torch.nn as nn


class Seq2Seq_Attn_Beam(nn.Module):
    def __init__(self):
        super(Seq2Seq_Attn_Beam, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder_Attn_Beam()


    def forward(self, input, target, input_length, target_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        decoder_output, decoder_hidden = self.decoder(encoder_hidden, target, encoder_output)
        return decoder_output


    def evaluation_attn(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        decoder_output, predict_result = self.decoder.evaluate(encoder_hidden, encoder_output)
        return decoder_output, predict_result


    def evaluation_beam(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        best_seq = self.decoder.evaluate_beam(encoder_hidden, encoder_output)
        return best_seq