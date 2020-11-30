

class Word2Sequence(object):
    PAD_TAG = '<PAD>'
    UNK_TAG = '<UNK>'
    SOS_TAG = '<SOS>'
    EOS_TAG = '<EOS>'

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3


    def __init__(self):
        # 词表字典
        self.dict = {
            self.PAD_TAG:self.PAD,
            self.UNK_TAG:self.UNK,
            self.SOS_TAG:self.SOS,
            self.EOS_TAG:self.EOS
        }
        # 统计词频用字典
        self.count = {}


    def fit(self, sentence):
        """
        统计每句话中的词频
        :param sentence: 经过分词后的句子
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1


    def build_vocab(self, min = 5, max = None, max_features = None):
        """
        构建词表
        :param min: 最小词频
        :param max: 最大词频
        :param max_features: 最多特征个数
        :return:
        """
        # 注意两个条件都有等号
        if min is not None:
            self.count = {k : v for k, v in self.count.items() if v >= min}
        if max is not None:
            self.count = {k : v for k, v in self.count.items() if v <= max}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), key = lambda x : x[1], reverse = True)[: max_features])
        # 构建词表
        for word in self.count:
            self.dict[word] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))


    def transfrom(self, sentence, max_len = None, add_eos = False):
        """
        文本转序列
        :param sentence: 分词后的句子
        :param max_len: 句子最大长度
        :param add_eos: 是否添加结束标记
        :return:
        """

        if max_len and add_eos:
            max_len = max_len -1
        # 句子过长进行裁剪
        if max_len <= len(sentence):
            sentence = sentence[:max_len]
        # 句子过短进行填充
        if max_len > len(sentence):
            sentence += [self.PAD_TAG] * (max_len - len(sentence))
        # 若添加结束标记
        if add_eos:
            # 在pad标记前添加EOS
            if self.PAD_TAG in sentence:
                index = sentence.index(self.PAD_TAG)
                sentence.insert(index, self.EOS_TAG)
            # 无pad的情况下，直接添加EOS
            else:
                sentence += [self.EOS_TAG]
        return [self.dict.get(i, self.UNK) for i in sentence]


    def inverse_transform(self, indices):
        """
        序列转文本
        :param indices: 序列
        :return:
        """
        result = []
        for i in indices:
            # 进行序列和文本的转化，若未知字符，采用UNK代替
            temp = self.inverse_dict.get(i, self.UNK_TAG)
            # 判断是否遇到结束标记EOS,若是结束添加
            if i != self.EOS_TAG:
                result.append(temp)
            else:
                break
        # 将转换好的文字进行拼接为一句话
        return ''.join(result)


    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    sentences = [["今天","天气","很","好"],
                  ["今天","去","吃","什么"]]
    ws = Word2Sequence()
    for sentence in sentences:
        ws.fit(sentence)
    ws.build_vocab(min = 1)
    print('vocab_dict', ws.dict)
    ret = ws.transfrom(["好","好","好","好","好","好","好","热","呀"],max_len=13, add_eos=True)
    print('transfrom',ret)
    ret = ws.inverse_transform(ret)
    print('inverse',ret)
    print(ws.PAD_TAG)
    print(ws.PAD)