import time
def cau_acc(pred, labels):
    '''
    Calculate the accuracy. 
    pred.shape = [batch_size]
    pred: the predict labels. 

    labels.shape = [batch_size]
    labels: the gold labels. 
    '''
    acc = 0.0
    wrong_ids = []
    for i in range(len(labels)):
        if labels[i] == pred[i]:
            acc += 1.0
        else:
            wrong_ids.append(i)
    acc /= len(labels)
    return acc, wrong_ids

def cau_samples_acc(samples):
    acc = 0.0
    for sample in samples:
        if sample.is_pred_right():
            acc += 1.0
    acc /= len(samples)
    return acc

def cau_recall_mrr(preds,labels,cutoff):
    recall = []
    mrr = []
    for batch, b_label in zip(preds,labels):
        for step, s_label in zip(batch,b_label):
            ranks = (step[s_label] < step).sum() +1
            recall.append(ranks<=cutoff)
            mrr.append(1/ranks if ranks <= cutoff else 0.0)
    return recall, mrr

def cau_recall_mrr_org(preds,labels,cutoff = 20):
    recall = []
    mrr = []
    rank_l = []

    for batch, b_label in zip(preds,labels):

        ranks = (batch[b_label] < batch).sum() +1
        rank_l.append(ranks)
        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)
    return recall, mrr, rank_l

def cau_recall_mrr_n(preds,labels,cutoff = 20):
    recall = []
    mrr = []
    for batch, b_label in zip(preds,labels):

        ranks = (batch[b_label] < batch).sum() +1

        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)
    return recall, mrr

def cau_samples_recall_mrr(samples, cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += sum(x <= cutoff for x in sample.pred)
        mrr += sum(1/x if x <= cutoff else 0 for x in sample.pred)
    num = 0
    for sample in samples:
        num += len(sample.pred)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr

def new_cau_samples_recall_mrr(samples,cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += (1 if sample.pred[0] <= cutoff else 0)
        mrr += (1/sample.pred[0] if sample.pred[0] <=cutoff else 0)
    num = len(samples)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr