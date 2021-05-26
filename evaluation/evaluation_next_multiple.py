import time
import numpy as np

def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'): 
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by
    1- HitRate@N and MRR@N (immediate next item recommendation task)
    2- Precision@N, Recall@N and MAP@N (multiple items recommendation task)
    Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    
    '''
    
    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')
    
    sc = time.clock();
    st = time.time();
    
    time_sum = 0
    time_sum_clock = 0
    time_count = 0
    
    for m in metrics:
        m.reset();
    
    test_data.sort_values([session_key, time_key], inplace=True)
    test_data = test_data.reset_index(drop=True)
    
    items_to_predict = train_data[item_key].unique()
    
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    length_session = np.zeros(test_data[session_key].nunique(), dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    length_session[0:] = test_data.groupby(session_key).size()
    
    current_session_idx = 0
    pos = offset_sessions[current_session_idx]
    position = 0
    finished = False
    
    while not finished:
        
        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')
            
        
        crs = time.clock();
        trs = time.time();
        
        current_item = test_data[item_key][pos]
        current_session = test_data[session_key][pos]
        ts = test_data[time_key][pos]
        rest = test_data[item_key][pos+1:offset_sessions[current_session_idx]+length_session[current_session_idx]].values
        
        for m in metrics:
            if hasattr(m, 'start_predict'):
                m.start_predict( pr )

        preds = pr.predict_next(current_session, current_item, items_to_predict, timestamp=ts)
        
        for m in metrics:
            if hasattr(m, 'stop_predict'):
                m.stop_predict( pr )
            
        preds[np.isnan(preds)] = 0
#         preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
        preds.sort_values( ascending=False, inplace=True )
        
        time_sum_clock += time.clock()-crs
        time_sum += time.time()-trs
        time_count += 1
        
        count += 1
        
        for m in metrics:
            if hasattr(m, 'add_multiple'):
                m.add_multiple( preds, rest, for_item=current_item, session=current_session, position=position)
            elif hasattr(m, 'add'):
                    m.add(preds, rest[0], for_item=current_item, session=current_session, position=position)
        
        pos += 1
        position += 1
        
        if pos + 1 == offset_sessions[current_session_idx]+length_session[current_session_idx]:
            current_session_idx += 1
            
            if current_session_idx == test_data[session_key].nunique():
                finished = True
            
            pos = offset_sessions[current_session_idx]
            position = 0
            count += 1
        
        
    
    print( 'END evaluation in ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )
    print( '    time count ', (time_count), 'count/', (time_sum), ' sum' )
    
    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append( m.result() )
    
    return res

