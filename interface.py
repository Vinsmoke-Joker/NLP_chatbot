from cut_sentence import cut
import torch
import config
from Seq2Seq import Seq2SeqModel
import os


# 模拟聊天场景，对用户输入进来的话进行回答
def interface():
    # 加载训练集好的模型
    model = Seq2SeqModel().to(config.device)
    assert os.path.exists('./model/chatbot_model.pkl') , '请先在train中对模型进行训练!'
    model.load_state_dict(torch.load('./model/chatbot_model.pkl'))
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
        outputs, predict = model.evaluation(input_tensor, input_length_tensor)
        # 进行序列转换文本
        result = config.target_ws.inverse_transform(predict[0])
        print('chatbot>>:', result)


if __name__ == '__main__':
    interface()