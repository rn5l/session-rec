import time
from pympler.asizeof import asizeof

class Time_usage_training:

    def __init__(self):
        self.start_time=0
        self.end_time=0

    def init(self, algorithm=None):
        self.start_time=0
        self.end_time=0
        self.start_time = time.time()
        
    def start(self, algorithm=None):
        self.start_time=0
        self.end_time=0
        self.start_time = time.time()

    def stop(self, algorithm=None):
        self.end_time=time.time()

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Training time:"), (self.end_time - self.start_time)

    def reset(self):
        pass
    
class Memory_usage:

    def __init__(self):
        self.memory=0

    def init(self, algorithm):
        self.memory=0
    
    def start(self, algorithm=None):
        self.memory=0
    
    def stop(self, algorithm):
        self.memory = asizeof( algorithm )
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Memory usage:"), (self.memory)

    def reset(self):
        pass

class Time_usage_testing:

    def __init__(self):
        pass

    def init(self, train):
        self.start_time = 0
        self.start_time_cpu = 0
        self.time_sum_cpu = 0
        self.time_sum = 0
        self.time_count = 0

    def start_predict(self, algorithm):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        self.start_time = time.time()
        self.start_time_cpu = time.clock();

    def stop_predict(self, algorithm):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        self.time_count += 1
        self.time_sum_cpu = time.clock() - self.start_time_cpu
        self.time_sum = time.clock() - self.start_time
    
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Predcition time:", "Predcition time CPU:"), (self.time_sum / self.time_count, self.time_sum_cpu / self.time_count)
    
    def result_second(self,second_time):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Testing time seconds:"), (second_time)

    def result_cpu(self,cpu_time):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Testing time cpu:"), (cpu_time)

    def reset(self):
        pass