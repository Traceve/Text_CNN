# coding: utf-8
#此为数据预处理文件
from collections import Counter

import jieba
import numpy as np
import tensorflow.contrib.keras as kr

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')#以UTF-8的格式代开文件并返回


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []#定义两个空列表
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')# 移除每行头尾空格或换行符，然后根据tab把label和content分到list里
                if content:
                    contents.append(jieba.lcut(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)#训练集读进来

    all_data = []#定义个空的集合
    for content in data_train:
        all_data.extend(content)#把训练集的内容加进去

    counter = Counter(all_data)#统计文档中每个字符出现的次数
    count_pairs = counter.most_common(vocab_size - 1)#挑出词汇表中字符个数出现最高的
    words, _ = list(zip(*count_pairs))# 格式：[('c', 'a'), (3, 1)], words格式为：('c', 'a')
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')#vocab_dir里面就是处理后的词表，每行一个字


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]#依次读取每行并去掉每行的的头空白得到词列表
    word_to_id = dict(zip(words, range(len(words))))#转换为{词，id}的形式
    return words, word_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):#实际就是将一篇文档的词id向量和一个分类id对应起来
        #data_id中每个元素是一篇文档的词id构成的向量
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # label_id， 每篇文档对应一个分类id，这个分类id是与一篇文档的词id向量对应
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    #因为data_id中每个元素都是一个由一篇文档中的字组成的向量，而每篇文档长度不同，所以每篇文档对应的向量元素个数不同，所以这里要将他们格式化为同一长度，策略就是高位补0]
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=128):
    """生成批次数据"""
    data_len = len(x)
    np.random.seed(1000)
    num_batch = int((data_len - 1) / batch_size) + 1#可以将语料分成多少个batch_size
    indices = np.random.permutation(np.arange(data_len))#做shuffle 即进行对语料的前后顺序进行打乱 在整个语料上进行打乱操作
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)#只有在最后一个batch上，才有可能取data_len，前面的都是取前者
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
