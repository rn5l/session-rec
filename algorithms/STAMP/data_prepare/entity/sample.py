#coding=utf-8

class Sample(object):
    '''
    一个样本。
    '''
    def __init__(self):
        self.id = -1
        self.session_id = -1
        self.click_items = []

        self.items_idxes = []

        self.in_idxes = []
        self.out_idxes = []
        self.label = []
        self.pred =[]
        self.best_pred = []
        self.ext_matrix = {'alpha':[]} # 额外数据，key是名字，value是矩阵。例如attention.

    def __str__(self):
        ret = 'id: ' + str(self.id) + '\n'
        ret += 'session_id: ' + str(self.session_id) + '\n'
        ret += 'items: '+ str(self.items_idxes) + '\n'
        ret += 'click_items: '+ str(self.click_items) + '\n'
        ret += 'out: ' + str(self.out_idxes) + '\n'
        ret += 'in: '+ str(self.in_idxes) + '\n'
        ret += 'label: '+ str(self.label) + '\n'
        return ret