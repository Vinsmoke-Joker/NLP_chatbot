import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloader
import config
import numpy as np
from Seq2Seq import Seq2SeqModel
import os
from tqdm import tqdm



model = Seq2SeqModel().to(config.device)
if os.path.exists('./model/chatbot_model.pkl'):
    model.load_state_dict(torch.load('./model/chatbot_model.pkl'))


def eval():
    model.eval()
    loss_list = []
    test_data_loader = get_dataloader(train = False)
    with torch.no_grad():
        bar = tqdm(test_data_loader, desc = 'testing', total = len(test_data_loader))
        for idx, (input, target, input_length, target_length) in enumerate(bar):
            input = input.to(config.device)
            target = target.to(config.device)
            input_length = input_length.to(config.device)
            target_length = target_length.to(config.device)
            # 获取模型的预测结果
            decoder_outputs, predict_result = model.evaluation(input, input_length)
            # 计算损失
            loss = F.nll_loss(decoder_outputs.view(-1, len(config.target_ws)), target.view(-1), ignore_index = config.target_ws.PAD)
            loss_list.append(loss.item())
            bar.set_description('idx{}:/{}, loss:{:.6f}'.format(idx, len(test_data_loader), np.mean(loss_list)))


if __name__ == '__main__':
    eval()