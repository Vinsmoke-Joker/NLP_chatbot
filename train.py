import torch
from Seq2Seq import Seq2SeqModel
import torch.optim as optim
import config
from dataset import get_dataloader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import os


model = Seq2SeqModel().to(config.device)
if os.path.exists('./model/chatbot_model.pkl'):
    model.load_state_dict(torch.load('./model/chatbot_model.pkl'))

optimizer = optim.Adam(model.parameters())
loss_list = []


def train(epoch):
    train_dataloader = get_dataloader(train = True)
    bar = tqdm(train_dataloader, desc = 'training', total = len(train_dataloader))
    model.train()
    for idx, (input, target, input_length, target_length) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_length = target_length.to(config.device)
        optimizer.zero_grad()
        decoder_outputs = model(input, target, input_length, target_length)
        # 由于在decoder中进行log_softmax计算，计算损失需要F.nll_loss
        # decoder_outputs [batch_size, seq_len, vocab_size]
        # target [batch_size, seq_len]
        loss = F.nll_loss(decoder_outputs.view(-1, len(config.target_ws)), target.view(-1), ignore_index = config.target_ws.PAD)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description('epoch:{}, idx{}/{}, loss:{:.6f}'.format(epoch + 1, idx, len(train_dataloader), np.mean(loss_list)))
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/chatbot_model.pkl')


if __name__ == '__main__':
    for i in range(100):
        train(i)