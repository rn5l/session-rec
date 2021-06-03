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
    if pr.support_users():
       if pr.predict_with_training_data():
            test_data = pd.concat([train_data, test_data])

    test_data.sort_values([user_key, session_key, time_key], inplace=True)
    prev_iid, prev_uid, prev_sid = -1, -1, -1
    pos = 0

    for i in range(len(test_data)):

        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', len(test_data), ' actions: ', ( count / len(test_data) * 100.0 ), ' % in',(time.time()-st), 's')


        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        uid = test_data[user_key].values[i] if user_key is not None else -1

        if prev_sid != sid:
            prev_sid = sid
            pos = 0
            if hasattr(pr, 'predict_for_extended_model'):
                past_items = pr.predict_for_extended_model(uid)
                for past_item in past_items:
                    # to update the state for the current session, we do not need the predictions
                    pr.predict_next(sid, past_item, uid, items_to_predict)
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))

            crs = time.clock();
            trs = time.time();

            if test_data['in_eval'].values[i] == True:
                for m in metrics:
                    if hasattr(m, 'start_predict'):
                        m.start_predict(pr)

            if pr.support_users():  # session-aware (e.g. hgru4rec)
                preds = pr.predict_next(sid, prev_iid, uid, items_to_predict)
            else: # session-based (e.g. sknn)
                preds = pr.predict_next(sid, prev_iid, items_to_predict)  # without user_id

            if test_data['in_eval'].values[i] == True:
                for m in metrics:
                    if hasattr(m, 'stop_predict'):
                        m.stop_predict(pr)

            preds[np.isnan(preds)] = 0
            # preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
            preds.sort_values(ascending=False, inplace=True)

            time_sum_clock += time.clock() - crs
            time_sum += time.time() - trs
            time_count += 1

            if test_data['in_eval'].values[i] == True:
                for m in metrics:
                    if hasattr(m, 'add'):
                        m.add(preds, iid, for_item=prev_iid, session=sid, position=pos)

            pos += 1

        prev_iid = iid

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
    pr.predict = None

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()
    # select the bootstrap_length recent sessions in training data to bootstrap the hidden state of the predictor
    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    test_data = pd.concat([train_data, test_data])
    # pre-process the session data
    user_indptr, offset_sessions = pr.preprocess_data(test_data)
    offset_users = offset_sessions[user_indptr]

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()
    other_columns.remove('in_eval')

    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0

    # here we use parallel minibatches over users
    if len(offset_users) - 1 < batch_size:
        batch_size = len(offset_users) - 1

    # variables used to iterate over users
    user_iters = np.arange(batch_size).astype(np.int32)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_iters]
    user_end = offset_users[user_iters + 1]

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    perc = 10
    n_users = len(offset_users)
    user_cnt = 0
    while True:

        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')

        # iterate only over the valid entries in the minibatch
        valid_mask = np.logical_and(user_iters >= 0, session_iters >= 0)
        if valid_mask.sum() == 0:
            break

        session_start_valid = session_start[valid_mask]
        session_end_valid = session_end[valid_mask]
        session_minlen = (session_end_valid - session_start_valid).min()
        in_item_id[valid_mask] = test_data[item_key].values[session_start_valid]
        in_user_id[valid_mask] = test_data[user_key].values[session_start_valid]
        in_session_id[valid_mask] = test_data[session_key].values[session_start_valid]

        for i in range(session_minlen - 1):
            out_item_idx = test_data[item_key].values[session_start_valid + i + 1]

            crs = time.clock();
            trs = time.time();

            for m in metrics: #TODO: must be started after preparing items_to_predict (when it's no None)
                if hasattr(m, 'start_predict'):
                    m.start_predict( pr )

            if items is not None:
                uniq_out = np.unique(np.array(out_item_idx, dtype=np.int32))
                preds = pr.predict_next(in_session_id, in_item_id, in_user_id, np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]))
                # preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id, np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]), batch_size)
            else:
                preds = pr.predict_next(in_session_id, in_item_id, in_user_id, None)
                # preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id, None, batch_size)

            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict( pr )

            preds.fillna(0, inplace=True)

            in_item_id[valid_mask] = out_item_idx
            in_eval_mask = np.zeros(batch_size, dtype=np.bool)
            in_eval_mask[valid_mask] = test_data['in_eval'].values[session_start_valid + i + 1]

            if np.any(in_eval_mask):
                if items is not None:
                    others = preds.ix[items].values.T[in_eval_mask].T
                    targets = np.diag(preds.ix[in_item_id].values)[in_eval_mask]
                    ranks = (others > targets).sum(axis=0) + 1
                else:
                    ranks = (preds.values.T[in_eval_mask].T > np.diag(preds.ix[in_item_id].values)[in_eval_mask]).sum(
                        axis=0) + 1

                for part, series in preds.loc[:, in_eval_mask].iteritems():
                    preds.sort_values(part, ascending=False, inplace=True)

                    time_sum_clock += time.clock() - crs
                    time_sum += time.time()-trs
                    time_count += 1

                    for m in metrics:
                        if hasattr(m, 'add'):
                            m.add(preds[part], in_item_id[part]) # add(result, next_item)

                rank_ok = ranks < cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / ranks[rank_ok]).sum()
                evalutation_point_count += len(ranks)

        session_start[valid_mask] = session_start[valid_mask] + session_minlen - 1
        session_start_mask = np.arange(len(user_iters))[valid_mask & (session_end - session_start <= 1)]
        for idx in session_start_mask:
            session_iters[idx] += 1
            if session_iters[idx] + 1 >= len(offset_sessions):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

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

        count += 1

    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )
    print( '    time count ', (time_count), 'count/', (time_sum), ' sum' )

    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
            pass
        else:
            res.append( m.result() )


    return res

def evaluate_sessions_batch_original(pr, train_data, test_data, items=None, cut_off=20, batch_size=100,
                                           break_ties=False,
                                           output_rankings=False,
                                           bootstrap_length=-1,
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
    # select the bootstrap_length recent sessions in training data to bootstrap the hidden state of the predictor
    if bootstrap_length >= 0:
        user_sessions = train_data.sort_values(by=[user_key, time_key])[[user_key, session_key]].drop_duplicates()
        session_order = user_sessions.groupby(user_key, sort=False).cumcount(ascending=False)
        last_sessions = user_sessions[session_order < bootstrap_length][session_key]
        train_data = train_data[train_data[session_key].isin(last_sessions)].copy()
    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    test_data = pd.concat([train_data, test_data])
    # pre-process the session data
    user_indptr, offset_sessions = pr.preprocess_data(test_data)
    offset_users = offset_sessions[user_indptr]

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()
    other_columns.remove('in_eval')

    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0

    if output_rankings:
        rank_list = []

    # here we use parallel minibatches over users
    if len(offset_users) - 1 < batch_size:
        batch_size = len(offset_users) - 1

    # variables used to iterate over users
    user_iters = np.arange(batch_size).astype(np.int32)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_iters]
    user_end = offset_users[user_iters + 1]

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    perc = 10
    n_users = len(offset_users)
    user_cnt = 0
    while True:
        # iterate only over the valid entries in the minibatch
        valid_mask = np.logical_and(user_iters >= 0, session_iters >= 0)
        if valid_mask.sum() == 0:
            break

        session_start_valid = session_start[valid_mask]
        session_end_valid = session_end[valid_mask]
        session_minlen = (session_end_valid - session_start_valid).min()
        in_item_id[valid_mask] = test_data[item_key].values[session_start_valid]
        in_user_id[valid_mask] = test_data[user_key].values[session_start_valid]
        in_session_id[valid_mask] = test_data[session_key].values[session_start_valid]

        for i in range(session_minlen - 1):
            out_item_idx = test_data[item_key].values[session_start_valid + i + 1] #out_item_idx: second item_id of every user_id (including all his sessions)
            if items is not None:
                uniq_out = np.unique(np.array(out_item_idx, dtype=np.int32))
                preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id,
                                              np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]),
                                              batch_size)
            else:
                preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id, None, batch_size)
            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8

            preds.fillna(0, inplace=True)

            in_item_id[valid_mask] = out_item_idx # next_items become as current_items (i.e. current_items become as previous_items)
            in_eval_mask = np.zeros(batch_size, dtype=np.bool)
            in_eval_mask[valid_mask] = test_data['in_eval'].values[session_start_valid + i + 1]

            if np.any(in_eval_mask):
                if items is not None:
                    others = preds.ix[items].values.T[in_eval_mask].T
                    targets = np.diag(preds.ix[in_item_id].values)[in_eval_mask]
                    ranks = (others > targets).sum(axis=0) + 1
                else:
                    ranks = (preds.values.T[in_eval_mask].T > np.diag(preds.ix[in_item_id].values)[in_eval_mask]).sum(
                        axis=0) + 1
                if output_rankings:
                    session_start_eval = session_start[in_eval_mask]
                    eval_record = [in_user_id[in_eval_mask],  # user id
                                   in_session_id[in_eval_mask],  # session id
                                   in_item_id[in_eval_mask],  # OUTPUT item id (see line 261)
                                   ranks]
                    others_record = np.vstack([test_data[c].values[session_start_eval + i + 1] for c in other_columns])
                    batch_results = np.vstack([eval_record, others_record]).T
                    rank_list.append(batch_results)

                rank_ok = ranks <= cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / ranks[rank_ok]).sum()
                evalutation_point_count += len(ranks)

        session_start[valid_mask] = session_start[valid_mask] + session_minlen - 1
        session_start_mask = np.arange(len(user_iters))[valid_mask & (session_end - session_start <= 1)]
        for idx in session_start_mask:
            session_iters[idx] += 1
            if session_iters[idx] + 1 >= len(offset_sessions):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

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

    if output_rankings:
        columns = [user_key, session_key, item_key, 'rank'] + other_columns
        df_ranks = pd.DataFrame.from_records(np.vstack(rank_list), columns=columns)
        return recall / evalutation_point_count, mrr / evalutation_point_count, df_ranks
    else:
        return recall / evalutation_point_count, mrr / evalutation_point_count

