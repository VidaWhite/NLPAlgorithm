#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Author: VidaWhite
# Source: 某网络课程稍作修改
# Date: 2018/4/13
# Description: 朴素贝叶斯做新闻主题分类

import os
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


# 制作词包
def make_word_set(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if word is not True or False and len(word > 0) and word not in words_set:
                words_set.add(word)
    return words_set


# 文本处理，也是样本生成的过程
def text_processing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 遍历文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 读取文件
        j = 1
        for file in files:
            if j > 100: #怕内存爆掉，只取100个样本文件，也可以注释掉取完
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
            # Jieba中文分词
            jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结果是一个可迭代的generator
            word_list = list(word_cut) # generator 转化为 list，每个词 unicode 格式
            jieba.disable_parallel() # 关闭并行分词模式

            data_list.append(word_list) # 训练集list
            class_list.append(folder.decode("utf-8")) # 类别
            j += 1

    # 手动划分训练集和测试集
    #data_class_list = zip(data_list, class_list)
    #random.shuffle(data_class_list)
    #index = int(len(data_class_list)*test_size) + 1
    #train_list = data_class_list[index:]
    #test_list = data_class_list[:index]
    #train_data_list, train_class_list = zip(*train_list)
    #test_data_list, test_class_list = zip(*test_list)

    # sklearn 划分训练集和测试集
    train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # key 函数利用词频降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) #内建函数sorted 参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


# 选取词频最高的1000个词作为特征词
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000: # feature_words的维度1000
            break

        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1

    return feature_words


# 文本特征
def text_features(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        if flag == 'nltk':
            # nltk 特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            # sklearn 特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []

        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]

    return train_feature_list, test_feature_list


# 分类， 同时输出准确率
def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    if flag == 'nltk':
        # 使用 nltk 分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        # sklearn 分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []

    return test_accuracy


if __name__ == '__main__':
    print "start"

    # 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path,
                                                                                                         test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = make_word_set(stopwords_file)

    # 文本特征提取和分类
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []

    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words, flag)
        test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)

    print test_accuracy_list

    # 结果评价
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()

    print "finish"
