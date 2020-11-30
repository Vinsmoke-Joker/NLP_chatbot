from Seq2Seq_attn import Seq2Seq_Attn
import torch
import torch.optim as optim
import os
from tqdm import tqdm
from dataset import get_dataloader
import config
import torch.nn.functional as F


model = Seq2Seq_Attn().to(config.device)
if os.path.exists('./model/chatbot_attn_model.pkl'):
    model.load_state_dict(torch.load('./model/chatbot_attn_model.pkl'))
optimizer = optim.Adam(model.parameters())
loss_list = []

def train_attn(epoch):
    model.train()
    train_dataloader = get_dataloader(train = True)
    bar = tqdm(train_dataloader, desc = 'attn_training...', total = len(train_dataloader))
    for idx, (input, target, input_length, target_length) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_length = target_length.to(config.device)
        optimizer.zero_grad()
        outputs = model(input, target, input_length, target_length)
        # outputs [batch_size, de_seq_len, vocab_size]
        # target [batch_size, de_seq_len]
        loss = F.nll_loss(outputs.view(-1, len(config.target_ws)), target.view(-1), ignore_index = config.target_ws.PAD)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description('epoch:{},idx:{}/{},loss{:.6f}'.format(epoch + 1, idx, len(train_dataloader), loss.item()))
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/chatbot_attn_model.pkl')


if __name__ == '__main__':
    for i in range(10):
        train_attn(i)