import string
import jieba
import jieba.posseg as psg
import logging


# 停用词和自定义词典路径
stopwords_path = './corpus/stopwords.txt'
keywords_path = './corpus/keywords.txt'
# 英文字母
letters = string.ascii_lowercase
# 关闭jieba的日志
jieba.setLogLevel(logging.INFO)
# 读取所有停用词到列表
stop_words = [i.strip() for i in open(stopwords_path, encoding = 'utf-8').readline()]


def cut(sentence, by_word = False, use_stopwords = False, use_seg = False):
    """
    分词方法
    :param sentence: 待分词的句子
    :param by_word: 按字切分
    :param use_stopwords: 使用停用词
    :param use_seg: 返回词性
    :return:
    """
    if by_word:
        return cut_sentence_by_word(sentence)
    else:
        return cut_sentence(sentence, use_stopwords, use_seg)


def cut_sentence(sentence, use_stopwords, use_seg):
    if use_seg:
        # 使用psg.lcut进行切分，返回[(i.word, i.flag)...]
        result = psg.lcut(sentence)
        if use_stopwords:
            result = [i for i in result if i[0] not in stop_words]
    else:
        result = jieba.lcut(sentence)
    return result


def cut_sentence_by_word(sentence):
    """
    按字进行切分
    :param sentence: 待分词的语句
    :return:
    """
    temp = '' # 用于拼接英文字符
    result = [] # 保存结果
    # 按字遍历
    for word in sentence:
        # 判断是否是英文字母
        if word in letters:
            temp += word
        # 若遇到非英文字符有两种情况
        # 1.temp = ''意味当前是个汉字,将word直接存入result中
        # 2.temp != '' 意味着拼接完了一个单词，遇到了当前汉字，需要将temp存入reslut,并置空
        else:
            if len(temp) > 0:
                result.append(temp)
                temp = ''
            result.append(word)
        # 当遍历完所有字后，最后一个字母可能存储在temp中
    if len(temp) > 0:
        result.append(temp)
    return result


if __name__ == '__main__':
    sentence = '今天天气好热a'
    res1 = cut(sentence, by_word = True)
    res2 = cut(sentence, by_word = False, use_seg = True)
    res3 = cut(sentence, by_word = False, use_seg = False)
    print(res1)
    print(res2)
    print(res3)