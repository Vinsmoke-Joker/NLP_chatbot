import random
from tqdm import tqdm
import config
import torch
from torch.utils.data import Dataset, DataLoader


class ChatbotDataset(Dataset):
    def __init__(self, train = True):
        """
        :param train: 指定生成训练数据还是测试数据
        """
        input_path = './corpus/train_input.txt' if train else './corpus/test_input.txt'
        target_path = './corpus/train_target.txt' if train else './corpus/test_target.txt'
        self.input_data = open(input_path, encoding = 'utf-8').readlines()
        self.target_data = open(target_path, encoding = 'utf-8').readlines()
        # 由于闲聊模型，因此输入必须对应一条输出
        assert len(self.input_data) == len(self.target_data), '输入输出长度不一致！'


    def __getitem__(self, idx):
        input = self.input_data[idx].strip().split()
        target = self.target_data[idx].strip().split()
        # 获取真实长度
        input_length = len(input) if len(input) < config.chatbot_input_max_len else config.chatbot_input_max_len
        target_length = len(target) if len(target) < config.chatbot_target_max_len else config.chatbot_target_max_len

        # 对文本进行序列转换
        input = config.input_ws.transfrom(input, max_len = config.chatbot_input_max_len)
        target = config.target_ws.transfrom(target, max_len = config.chatbot_target_max_len)
        return input, target, input_length, target_length

    
    def __len__(self):
        return len(self.input_data)


def get_dataloader(train = True):
    # 获取训练集和测试集的dataloader
    batch_size = config.chatbot_train_batch_size if train else config.chatbot_test_batch_size
    return DataLoader(ChatbotDataset(train), batch_size = batch_size, shuffle = True, collate_fn = collate_fn)


def collate_fn(batch):
    # 需要对每个batch按长度进行排序
    batch = sorted(batch, key = lambda x : x[2], reverse = True)
    input, target, input_length, target_length = zip(*batch)
    # 封装为tensor
    input_tensor = torch.LongTensor(input)
    target_tensor = torch.LongTensor(target)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input_tensor, target_tensor, input_length, target_length 


if __name__ == '__main__':
    train_data_loader = get_dataloader(train = False)
    for idx, (input, target, input_length, target_length) in enumerate(train_data_loader):
        print(input)
        print(input.size())
        print(target)
        print(target.size())
        print(input_length)
        print(input_length.size())
        print(target_length)
        print(target_length.size())
        break
    print(config.target_ws.dict)
    print(len(config.input_ws))
    print(len(config.target_ws))