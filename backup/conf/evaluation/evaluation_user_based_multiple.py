import time
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def evaluate_sessions(pr, metrics, test_data, train_data, items=None, session_key='SessionId',
                                 user_key='UserId', item_key='ItemId', time_key='Time'):
    """
    Evaluates the HGRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Concatenates train sessions to test sessions to bootstrap the hidden states of the HGRU.
    The number of the last sessions of each user that are used in the bootstrapping is controlled by `bootstrap_length`.

    Parameters
    --------
    pr : gru4rec.HGRU4Rec
        A trained instance of the HGRU4Rec network.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs,
        one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to.
        If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Default value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation.
         If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties,
        which can mess up the evaluation.
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates;
        (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    bootstrap_length: int
        Number of sessions in train data used to bootstrap the hidden state of the predictor,
        starting from the last training session of each user.
        If -1, consider all sessions. (default: -1)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """

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

    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    # pr.predict = None

    items_to_predict = train_data[item_key].unique()

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()

    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    if pr.support_users(): # e.g. hgru4rec
        if pr.predict_with_training_data():
            test_data = pd.concat([train_data, test_data])

    test_data.sort_values([user_key, session_key, time_key], inplace=True)
    test_data = test_data.reset_index(drop=True)

    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    length_session = np.zeros(test_data[session_key].nunique(), dtype=np.int32)
    offset_sessions[1:] = test_data.groupby([user_key, session_key]).size().cumsum() # offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    length_session[0:] = test_data.groupby([user_key, session_key]).size() # length_session[0:] = test_data.groupby(session_key).size()

    current_session_idx = 0
    # pos: to iterate over test data to retrieve the current session and it's first interaction
    pos = offset_sessions[current_session_idx] # index of the first element of the current session in the test data
    position = 0  # position (index) of the current element in the current session
    finished = False

    prev_sid = -1
    while not finished:

        if count % 1000 == 0:
            print('    eval process: ', count, ' of ', len(test_data), ' actions: ', (count / len(test_data) * 100.0), ' % in',
                  (time.time() - st), 's')


        crs = time.clock();
        trs = time.time();

        current_item = test_data[item_key][pos]
        current_session = test_data[session_key][pos]
        current_user = test_data[user_key][pos] # current_user = test_data[user_key][pos] if user_key is not None else -1
        ts = test_data[time_key][pos]
        rest = test_data[item_key][
               pos + 1:offset_sessions[current_session_idx] + length_session[current_session_idx]].values

        if prev_sid != current_session:
            prev_sid = current_session
            if hasattr(pr, 'predict_for_extended_model'):
                past_items = pr.predict_for_extended_model(current_user)
                for past_item in past_items:
                    pr.predict_next(current_session, past_item, current_user, items_to_predict)  # to update the state for the current session, we do not need the predictions

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict(pr)

        if pr.support_users():  # session-aware (e.g. hgru4rec)
            preds = pr.predict_next(current_session, current_item, current_user, items_to_predict, timestamp=ts)
        else:  # session-based (e.g. sknn)
            preds = pr.predict_next(current_session, current_item, items_to_predict, timestamp=ts)  # without user_id

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict(pr)

        preds[np.isnan(preds)] = 0
         #  preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
        preds.sort_values(ascending=False, inplace=True)

        time_sum_clock += time.clock() - crs
        time_sum += time.time() - trs
        time_count += 1

        count += 1

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'add_multiple'):
                    m.add_multiple(preds, rest, for_item=current_item, session=current_session, position=position)

        pos += 1
        position += 1

        # check if we make prediction for all items of the current session (except the last one)
        if pos + 1 == offset_sessions[current_session_idx] + length_session[current_session_idx]:
            current_session_idx += 1 # start the next session

            if current_session_idx == test_data[session_key].nunique(): # if we check all sessions of the test data
                finished = True # finish the evaluation

            # retrieve the index of the first interaction of the next session we want to iterate over
            pos = offset_sessions[current_session_idx]
            position = 0 # reset the first position of the first interaction in the session
            # increment count because of the last item of the session (which we do not make prediction for)
            count += 1


    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
    print('    time count ', (time_count), 'count/', (time_sum), ' sum')

    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock / time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append(m.result())

    return res


def evaluate_sessions_hgru(pr, metrics, test_data, train_data, items=None, cut_off=20, batch_size=1,
                                 session_key='SessionId', user_key='UserId', item_key='ItemId',
                                 time_key='Time'):
    """
    Evaluates the HGRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Concatenates train sessions to test sessions to bootstrap the hidden states of the HGRU.
    The number of the last sessions of each user that are used in the bootstrapping is controlled by `bootstrap_length`.

    Parameters
    --------
    pr : gru4rec.HGRU4Rec
        A trained instance of the HGRU4Rec network.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs,
        one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to.
        If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Default value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation.
         If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties,
        which can mess up the evaluation.
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates;
        (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    bootstrap_length: int
        Number of sessions in train data used to bootstrap the hidden state of the predictor,
        starting from the last training session of each user.
        If -1, consider all sessions. (default: -1)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """

    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    pr.predict = None

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()
    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    test_data = pd.concat([train_data, test_data])

    # pre-process the session data
    # user_indptr, offset_sessions = pr.preprocess_data(test_data)
    # code inside preprocess_data --- START
    test_data.sort_values([user_key, session_key, time_key], inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    offset_sessions = np.r_[0, test_data.groupby([user_key, session_key], sort=False).size().cumsum()[:-1]]
    user_indptr = np.r_[0, test_data.groupby(user_key, sort=False)[session_key].nunique().cumsum()[:-1]]
    # code inside preprocess_data --- END
    offset_users = offset_sessions[user_indptr]

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()
    other_columns.remove('in_eval')

    # evalutation_point_count = 0
    # mrr, recall = 0.0, 0.0

    # if output_rankings:
    #     rank_list = []

    # here we use parallel minibatches over users
    if len(offset_users) - 1 < batch_size:
        batch_size = len(offset_users) - 1

    # variables used to iterate over users
    user_iters = np.arange(batch_size).astype(np.int32)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_iters] # index of the first session of the (current) user
    user_end = offset_users[user_iters + 1] # index of the first session of the next user

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters] # index of the first interaction of the (current) session
    session_end = offset_sessions[session_iters + 1] # index of the first interaction of the next session

    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)
    # in_ts = np.zeros(batch_size, dtype=np.int32) #SARA

    np.random.seed(42)
    perc = 10
    n_users = len(offset_users) # total number of users
    user_cnt = 0 # counter for number of users

    # for evaluation_multiple --- START

    count = 0
    # print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock();
    st = time.time();

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset();

    items_to_predict = train_data[item_key].unique()

    # offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    length_session = np.zeros(test_data[session_key].nunique(), dtype=np.int32)
    # offset_sessions[1:] = test_data.groupby(session_key).size().cumsum() # Duplicated! It is calculated in
    # length_session[0:] = test_data.groupby([user_key, session_key], sort=False).size()
    length_session[0:] = np.r_[test_data.groupby(session_key).size()]

    current_session_idx = 0
    # pos: to iterate over test data to retrieve the current session and it's first interaction
    pos = offset_sessions[current_session_idx]  # index of the first element of the current session in the test data
    position = 0  # position (index) of the current element in the current session
    # for evaluation_multiple --- END

    while True:

        # iterate only over the valid entries in the minibatch
        valid_mask = np.logical_and(user_iters >= 0, session_iters >= 0)
        if valid_mask.sum() == 0:
            break # end of the "while True"

        session_start_valid = session_start[valid_mask] # ndarray - first index of the current session(s)
        session_end_valid = session_end[valid_mask] # ndarray - first index of the next session(s)
        session_minlen = (session_end_valid - session_start_valid).min() # int64 - length of the shortest session in the batch
        in_item_id[valid_mask] = test_data[item_key].values[session_start_valid] # ndarray - current item id
        in_user_id[valid_mask] = test_data[user_key].values[session_start_valid] # ndarray - current user id
        in_session_id[valid_mask] = test_data[session_key].values[session_start_valid] # ndarray - current session id


        for i in range(session_minlen - 1): # iterate over items of the session(s)
            out_item_idx = test_data[item_key].values[session_start_valid + i + 1] # ndarray


            crs = time.clock();
            trs = time.time();

            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict( pr )


            preds = pr.predict_next(in_session_id, in_item_id, in_user_id, None) # DataFrame TODO: do prediction just for 'in_eval = True', and not for all!

            # for evaluation_multiple --- START
            count += 1

            pos += 1
            position += 1
            # for evaluation_multiple --- END

            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict( pr )

            # if break_ties:
            #     preds += np.random.rand(*preds.values.shape) * 1e-8

            preds.fillna(0, inplace=True)

            in_item_id[valid_mask] = out_item_idx # ndarray - set next item as current item (to continue predicting for it)
            in_eval_mask = np.zeros(batch_size, dtype=np.bool) # ndarray
            in_eval_mask[valid_mask] = test_data['in_eval'].values[session_start_valid + i + 1] # ndarray - check if the "target item" is in the test data (or train data)

            if np.any(in_eval_mask): # if (any of) the target item(s) is in the test data, evaluate the prediction
                for part, series in preds.loc[:, in_eval_mask].iteritems():
                    preds.sort_values(part, ascending=False, inplace=True)

                    time_sum_clock += time.clock() - crs
                    time_sum += time.time()-trs
                    time_count += 1

                    # count += 1

                    rest = test_data[item_key][pos:offset_sessions[current_session_idx] + length_session[current_session_idx]].values # just work for batch = 1 -TODO: if want to work for batch as well, these two should be arrays: pos, current_session_idx

                    for m in metrics:
                        if hasattr(m, 'add_multiple'):
                            m.add_multiple(preds[part], rest)  # rest: the rest items of the session (which should be predicted)
                # for evaluation_multiple --- END


            # for evaluation_multiple --- START
            # check if we make prediction for all items of the current session (except the last one)
            if pos + 1 == offset_sessions[current_session_idx] + length_session[current_session_idx]:
                current_session_idx += 1  # start the next session

                if current_session_idx == test_data[
                    session_key].nunique():  # if we check all sessions of the test data
                    finished = True  # finish the evaluation

                # retrieve the index of the first interaction of the next session we want to iterate over
                pos = offset_sessions[current_session_idx]
                position = 0  # reset the first position of the first interaction in the session
                count += 1  # increment count because of the last item of the session (which we do not make prediction for)
            # for evaluation_multiple --- END


        session_start[valid_mask] = session_start[valid_mask] + session_minlen - 1
        session_start_mask = np.arange(len(user_iters))[valid_mask & (session_end - session_start <= 1)]
        for idx in session_start_mask:
            session_iters[idx] += 1
            if session_iters[idx] + 1 >= len(offset_sessions):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            session_start[idx] = offset_sessions[session_iters[idx]] # update the index of the first interaction of the (current) session
            session_end[idx] = offset_sessions[session_iters[idx] + 1] # update the index of the first interaction of the next session

        user_change_mask = np.arange(len(user_iters))[valid_mask & (user_end - session_start <= 0)]
        for idx in user_change_mask:
            user_cnt += 1
            if user_cnt > int(perc * n_users / 100):
                logger.info('User {}/{} ({}% completed)'.format(user_cnt, n_users, perc))
                perc += 10
            user_maxiter += 1
            if user_maxiter + 1 >= len(offset_users):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            user_iters[idx] = user_maxiter
            user_start[idx] = offset_users[user_maxiter]
            user_end[idx] = offset_users[user_maxiter + 1]
            session_iters[idx] = user_indptr[user_maxiter]
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]


    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
            pass
        else:
            res.append( m.result() )


    return res

