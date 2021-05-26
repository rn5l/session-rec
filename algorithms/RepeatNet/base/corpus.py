import codecs
import os
import pickle
import ast
from copy import copy

def load_item(file,session_key,item_key):
    item2id = {}
    id2item = {}
    #with codecs.open(file, encoding='utf-8') as f:
    id = 0
    for index, row in file.iterrows():
        name = int(row[item_key])
        if name not in item2id.keys():
            item2id[name] = id
            id2item[id] = name
            id += 1

    print('item size: ', len(item2id), len(id2item))
    return item2id,id2item

class SessionCorpus:
    def __init__(self, data,item2id):
        self.dataset = []
        self.data=data

        self.item2id=item2id

    def load(self,session_key,item_key):
        count=0
        # with codecs.open(self.file_path, encoding='utf-8') as f:
        #     for line in f:
        previous_items=[]
        session_id_old=""

        for index, row in self.data.iterrows():
            #lines = row.strip('\n').strip('\r').split('\t')
            #input=ast.literal_eval(lines[0])
            if int(row[item_key]) not in self.item2id.keys():
                print("not in train")
            if row[session_key]==session_id_old:
                input=copy(previous_items)
                output=[self.item2id[int(row[item_key])]]
                self.dataset.append([input,output])
                previous_items.append(self.item2id[int(row[item_key])])
            else:
                previous_items=[]
                input = self.item2id[int(row[item_key])]
                previous_items.append(input)
            session_id_old=row[session_key]
            count+=1
        print('data size: ',len(self.dataset))
        # self.dataset=sorted(self.dataset, key=lambda s: len(s[0]))
        return self.dataset