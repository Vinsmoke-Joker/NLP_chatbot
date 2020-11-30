from Seq2Seq_attn import Seq2Seq_Attn
import torch
import config
import numpy as np
from dataset import get_dataloader
import torch.nn.functional as F
from tqdm import tqdm


model = Seq2Seq_Attn().to(config.device)
model.load_state_dict(torch.load('./model/chatbot_attn_model.pkl'))
loss_list = []


def eval():
    model.eval()
    test_dataloader = get_dataloader(train = False)
    bar = tqdm(test_dataloader, desc = 'attn_test...', total = len(test_dataloader))
    for idx, (input, target, input_length, target_length) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_length = target_length.to(config.device)
        with torch.no_grad():
            output, predict_result = model.evaluation_attn(input, input_length)
            loss = F.nll_loss(output.view(-1, len(config.target_ws)), target.view(-1), ignore_index = config.target_ws.PAD)
            loss_list.append(loss.item())
            bar.set_description('idx:{}/{}, loss:{}'.format(idx, len(test_dataloader), np.mean(loss_list)))


if __name__ == '__main__':
    eval()