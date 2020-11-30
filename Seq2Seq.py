import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, target)
        return decoder_outputs


    def evaluation(self, input, input_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        decoder_outputs, predict_result = self.decoder.evaluate(encoder_hidden)
        return decoder_outputs, predict_result
