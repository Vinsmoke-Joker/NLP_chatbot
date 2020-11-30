from cut_sentence import cut
import torch
import config
from Seq2Seq_beam import Seq2Seq_Attn_Beam
import os


# 模拟聊天场景，对用户输入进来的话进行回答
def interface():
    # 加载训练集好的模型
    model = Seq2Seq_Attn_Beam().to(config.device)
    assert os.path.exists('./model/chatbot_attn_model.pkl') , '请先在train中对模型进行训练!'
    model.load_state_dict(torch.load('./model/chatbot_attn_model.pkl'))
    model.eval()

    while True:
        # 输入进来的原始字符串,进行分词处理
        input_string = input('me>>:')
        if input_string == 'q':
            print('下次再聊')
            break
        input_cuted = cut(input_string, by_word = True)
        # 进行序列转换和tensor封装
        input_tensor = torch.LongTensor([config.input_ws.transfrom(input_cuted, max_len = config.chatbot_input_max_len)]).to(config.device)
        input_length_tensor = torch.LongTensor([len(input_cuted)]).to(config.device)
        # 获取预测结果
        predict = model.evaluation_beam(input_tensor, input_length_tensor)
        # 进行序列转换文本
        # beam_search中 返回本身为一个序列列表
        result = config.target_ws.inverse_transform(predict)
        print('chatbot>>:', result)


if __name__ == '__main__':
    interface()