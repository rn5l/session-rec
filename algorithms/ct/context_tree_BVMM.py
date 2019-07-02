from collections import deque
import math

class StdExpert(object):
    def __init__(self, depth):
        self.weight = 1.0 / pow(2.0, depth)
        self.counts = {}
        self.total = 1.0
        
    def get_proba(self, item):
        c = self.counts.get(item, 0.0)
        return (1.0+c) / (1.0+self.total)
        
    def update(self, item):
        self.total += 1
        self.counts[item] = self.counts.get(item, 0) + 1
        
        
class DirichletExpert(StdExpert):
    def __init__(self, depth, nb_symbols=1.0):
        StdExpert.__init__(self, depth) 
        self.nb_symbols = nb_symbols
        
    def get_proba(self, item):
        c = self.counts.get(item, 0.0)
        return (c + 1) / (self.total + self.nb_symbols);
        

class BayesianMixtureExpert(StdExpert):
    def __init__(self, depth, experts):
        StdExpert.__init__(self, depth) 
        self.probabilities = [1.0/float(len(experts)) for exp in experts]
        self.experts = experts
        
    def get_proba(self, item):
        p = 0.0
        for proba, expert in zip(self.probabilities, self.experts):
            p += proba * expert.get_proba(item)
        return p 
    
    def bayesian_update(self, item):
        old_probs = []
        z = 0.0;
        for i, expert in enumerate(self.experts):
            proba = expert.get_proba(item)
            old_probs.append(proba)
            z += proba * self.probabilities[i]

        for i, old in enumerate(old_probs):
            self.probabilities[i] *= old / z;
            
    def update(self, item):
        for expert in self.experts:
            expert.update(item)
        self.bayesian_update(item)
        self.total += 1

class History(object):
    def __init__(self, maxlen):
        self.histories = {}
        self.maxlen = maxlen
        
    def get_history(self, user):
        history = self.histories.get(user)
        if history is None:
            history = deque(maxlen=self.maxlen)
            self.histories[user] = history
        return history

class TreeNode(object):
    def __init__(self, expert):
        self.expert = expert
        self.children = {}
        
    def get_child(self, item):
        ret = self.children.get(item)
        return ret
        
    def add_child(self, item, expert):
        child = TreeNode(expert)
        self.children[item] = child
        return child
        
    def get_depth(self):
        subs = [sub.get_depth() for sub in self.children.values()]
        return 1 + max(subs) if len(subs) > 0 else 0
        
    def get_nb_nodes(self):
        subs = [sub.get_nb_nodes() for sub in self.children.values()]
        return 1 + sum(subs)
        
class TreeRoot(TreeNode):
    def __init__(self, expert_constructor):
        root_expert = expert_constructor(0)
        TreeNode.__init__(self, root_expert)
        self.expert_constructor = expert_constructor
        
    def expand(self, history):
        ret = [self]
        node = self
        depth = 0
        for item in history:
            child = node.get_child(item)
            depth += 1
            if child is None:
                expert = self.expert_constructor(depth)
                child = node.add_child(item, expert)
            node = child
            
    def get_nodes(self, history):
        ret = [self]
        node = self
        for item in history:
            node = node.get_child(item)
            if node is None:
                break
            ret.append(node)
        return ret
        
    def get_n_most_probable(self, candidates, history):
        item_to_q = {}
        nodes = self.get_nodes(history)
        for item in candidates:
            q = 0.0
            for node in nodes:
                p = node.expert.get_proba(item)
                w = node.expert.weight
                q = w*p + (1.0-w)*q
            item_to_q[item] = q
        items_and_qs = item_to_q.items()

        return items_and_qs
        
    def update(self, item, history):
        q = 0.0
        nodes = self.get_nodes(history)
        for node in nodes:
            p = node.expert.get_proba(item)
            w = node.expert.weight
            q = w*p + (1.0-w)*q
            node.weight = w*p / q
            node.expert.update(item)
