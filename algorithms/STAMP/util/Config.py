# coding=utf-8
import os
import json
import copy
import re

# 去除空白符
p = re.compile('\s+')
def move_space(str):
    return re.sub(p, '', str)

def read_conf(model, path):
    '''
    model: 需要加载参数的模型
    path: 配置文件路径，可以是绝对路径，也可以是相对路径（以项目根目录开始）。
    '''
    cpath = copy.deepcopy(path)
    if not cpath.startswith("/"):
        cpath = os.path.abspath(".") + "/" + cpath

    def move_anno(line):
        if line.startswith('#'):  # 注释行
            return ""
        ls = line.split("#")
        if len(ls) == 1:    # 不带 #
            return line.strip()
        count = 0
        nl = ''
        for l in ls:
            count += l.count('"')
            nl += l
            if count % 2 == 0:
                break
            nl += '#'
        return nl.strip()

    def is_model(line):
        if line.startswith('[') and line.endswith(']'):
            return True
        else:
            return False
    
    def load_conf(path, models):

        f = open(path,encoding='UTF-8')
        rets = {}
        conf_str = ''
        model_found = False
        now_model = None
        parents = None
        ll = ''
        for line in f:
            line = line.strip()
            if line.endswith('\\'):
                ll += line[:-1]
                continue
            else:
                ll += line
            l = move_anno(ll)
            ll = ''
            if model_found:
                conf_str += l
                if l.endswith('}'):
                    conf = None
                    if conf_str != '':
                        print(str(conf_str))
                        conf = json.loads(conf_str)
                    model_param = {}
                    model_param['config'] = conf
                    model_param['parents'] = parents
                    rets[now_model] = model_param
                    conf_str = ''
                    parents = None
                    now_model = None
                    model_found = False
                    if len(rets) == len(models):
                        break

            else:
                l = move_space(l)
                if is_model(l):
                    l = l[1:][:-1]
                    mdls = l.split(':')
                    now_model = mdls[0]
                    if now_model in models:
                        model_found = True
                        if len(mdls) > 1:
                            parents = mdls[1:]
                            parents.reverse()
        return rets 

    models = None 
    mdl = None
    if isinstance(model, str) or isinstance(model, unicode):
        mdl = model
        models = [model]
    elif isinstance(model, list):
        if len(model) > 1:
            raise Exception(u"目前只支持加载一个模型的参数。", model)
        mdl = model[0]
        models = model
    else:
        raise Exception(u"错误的类型，应该是str或list。", model)

    rets = load_conf(cpath, models)[mdl]
    mconf = rets['config']
    parents = rets['parents']
    retconf = {}
    if parents is not None:
        rets = load_conf(cpath, parents)
        for now_model in parents:
            for (k,v) in rets[now_model]['config'].items():
                retconf[k] = v 
        for (k, v) in mconf.items():
            retconf[k] = v
    else:
        retconf = mconf
    return retconf
