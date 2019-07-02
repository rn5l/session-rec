# coding=utf-8
import time
import pickle as cp
import numpy as np
import sys, os

def TIPrint(samples, config, acc = {},print_att = False,Time = None):
    base_path = os.path.realpath(__file__)
    bps = base_path.split("/")[1:-2]
    base_path = "/"
    for bp in bps:
        base_path += bp + "/"
    base_path += 'output/'
    if Time is None:
        suf = time.strftime("%Y%m%d%H%M", time.localtime())
    else:
        suf = Time

    path = base_path + "text/" + config['model'] + "-" + config['dataset'] + "-" + suf + '.out'
    print_txt(path, samples, config, acc, print_att)
    return suf

def print_txt(path, samples, config,  acc = {}, print_att = False):
    '''
    写入文本数据，使用writer
    :param samples: 样本
    :param config: 模型参数
    :param acc: acc = {'max_acc':0.0, 'max_train_acc': 0.0}
    :return: None
    '''
    outfile = open(path, 'w')
    outfile.write('accuracy:\n')
    for k,v in acc.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nconfig:\n")
    for k,v in config.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nsample:\n")
    for sample in samples:
        outfile.write("id      :\t" + str(sample.id) + '\n')
        outfile.write("session    :\t" + str(sample.session_id) + '\n')
        outfile.write("in_items  :\t" + str(sample.in_idxes) + '\n')
        outfile.write("out_items  :\t" + str(sample.out_idxes) + '\n')
        outfile.write("predict :\t" + str(sample.best_pred) + '\n')
        if print_att:
            for ext_key in sample.ext_matrix:
                matrixs = sample.ext_matrix[ext_key]
                outfile.write("attention :\t" + str(ext_key) + '\n')
                matrix=matrixs[-1]
                for i in range(len(sample.in_idxes)):
                    outfile.write(str(sample.in_idxes[i]) + " :\t")
                    for att in matrix:
                        outfile.write(str(att[i]) + " ")
                    outfile.write("\n")
        outfile.write("\n")
    outfile.close()

def print_binary(path, datas):
    '''
    写入序列数据，用cPickle
    :param ids: 样本的id，需要写入文件系统的数据的id
    :param datas: datas = {'':[[]], ...}, [[]] 的第0个维度给出id. 需要根据ids从中挑选需要写入的数据重新构建字典
    :return: None
    '''
    dfile = open(path, 'w')
    cp.dump(datas, dfile)
    dfile.close()
    pass

