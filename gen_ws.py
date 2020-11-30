import pickle
from tqdm import tqdm
import random
import config
from word2sequence import Word2Sequence

def chatbot_data_split():
    # 对数据集进行切分
    input_ = open('./corpus/input.txt', encoding = 'utf-8').readlines()
    target = open('./corpus/target.txt', encoding = 'utf-8').readlines()
    # 训练集
    f_train_input = open('./corpus/train_input.txt', 'a', encoding = 'utf-8')
    f_train_target = open('./corpus/train_target.txt', 'a', encoding = 'utf-8')
    # 测试集
    f_test_input = open('./corpus/test_input.txt', 'a', encoding = 'utf-8')
    f_test_target = open('./corpus/test_target.txt', 'a', encoding  = 'utf-8')
    # 从input_ 和 target中每次各取一条数据，分别按8:2写入训练集和测试集
    for input_, target in tqdm(zip(input_, target), desc='spliting'):
        if random.random() > 0.2:
            f_train_input.write(input_)
            f_train_target.write(target)
        else:
            f_test_input.write(input_)
            f_test_target.write(target)

    f_train_input.close()
    f_train_target.close()
    f_test_input.close()
    f_test_target.close()


def gen_ws(train_path, test_path, save_path):
    """
    生成词表
    :param train_path: 训练数据
    :param test_path: 测试数据
    :param save_path:  保存路径
    :return:
    """
    ws = Word2Sequence()
    for line in tqdm(open(train_path, encoding = 'utf-8').readlines(), desc = 'build_vocab1..'):
        ws.fit(line.strip().split())
    for line in tqdm(open(test_path, encoding= 'utf-8').readlines(), desc = 'build_vocab2..'):
        ws.fit(line.strip().split())

    ws.build_vocab(min = 5, max = None, max_features = 5000)
    print(len(ws))
    pickle.dump(ws, open(save_path, 'wb'))


if __name__ == '__main__':
    chatbot_data_split()
    train_input_path = './corpus/train_input.txt'
    test_input_path = './corpus/test_input.txt'
    train_target_path = './corpus/train_target.txt'
    test_target_path = './corpus/test_target.txt'
    input_ws_path = './model/ws_input.pkl'
    target_ws_path = './model/ws_target.pkl'
    gen_ws(train_input_path, test_input_path, input_ws_path)
    gen_ws(train_target_path, test_target_path, target_ws_path)
