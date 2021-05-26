from _operator import itemgetter
from math import sqrt
import random
import time

import numpy as np
import pandas as pd
from math import log10
from datetime import datetime as dt
from datetime import timedelta as td


class UVMContextKNN:
    '''
    VMContextKNN( k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    weighting_time : boolean
        Experimental function to give less weight to items from older sessions (default: False)
    dwelling_time : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    last_n_days : int
        Use only data from the last N days. (default: None)
    last_n_clicks : int
        Use only the last N clicks of the current session when recommending. (default: None)
    extend : bool
        Add evaluated sessions to the maps.
    normalize : bool
        Normalize the scores in the end.
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    user_key : string
        Header of the user ID column in the input file. (default: 'UserId')

    extend_session_length: int
        extend the current user's session

    extending_mode: string
        how extend the user session (default: lastViewed)
        lastViewed: extend the current user's session with the his/her last viewed items #TODO: now it saves just X last items, and they might be exactly the same: can try as well: save 5 distinctive items
        score_based: higher score: if the items appeared in more previous sessions AND more recently #TODO

    boost_own_sessions: double
        to increase the impact of (give weight more weight to) the sessions which belong to the user. (default: None)
        the value will be added to 1.0. For example for boost_own_sessions=0.2, weight will be 1.2

    past_neighbors: bool
        Include the neighbours of the past user's similar sessions (to the current session) as neighbours (default: False)

    reminders: bool
        Include reminding items in the (main) recommendation list. (default: False)

    remind_strategy: string
        Ranking strategy of the reminding list (recency, session_similarity). (default: recency)

    remind_sessions_num: int
        Number of the last user's sessions that the possible items for reminding are taken from (default: 6)

    reminders_num: int
        length of the reminding list (default: 3)

    remind_mode: string
        The postion of the remining items in recommendation list (top, end). (default: end)

    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div',
                 dwelling_time=False, last_n_days=None, last_n_clicks=None, weighting_score='div', weighting_time=False,
                 normalize=True, idf_weighting=False, idf_weighting_session=False,
                 remind=True, push_reminders=False, add_reminders=False, extend=False, extending_mode='lastViewed',
                 extend_session_length=None, boost_own_sessions=None, past_neighbors=False,
                 reminders=False, remind_strategy='recency', remind_sessions_num=6, reminders_num=3, remind_mode='end',
                 session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId'):

        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting
        self.dwelling_time = dwelling_time
        self.weighting_score = weighting_score
        self.weighting_time = weighting_time
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.idf_weighting = idf_weighting
        self.idf_weighting_session = idf_weighting_session
        self.normalize = normalize
        self.last_n_days = last_n_days
        self.last_n_clicks = last_n_clicks  # user_based
        self.user_key = user_key  # user_based
        self.remind = remind  # True/False. If False: items form the current session will be excluded
        self.push_reminders = push_reminders
        self.add_reminders = add_reminders
        self.extend = extend
        self.extend_session_length = extend_session_length  # user_based
        self.extending_mode = extending_mode
        self.boost_own_sessions = boost_own_sessions  # user_based
        self.past_neighbors = past_neighbors  # user_based
        self.reminders = reminders  # reminders
        self.remind_strategy = remind_strategy  # reminders
        self.remind_sessions_num = remind_sessions_num  # reminders
        self.reminders_num = reminders_num  # reminders
        self.remind_mode = remind_mode  # reminders
        self.extend_session = False  # if we add more items to the session, we will assign it True, to find also relevant sessions for added items

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.items_previous = []  # user_based
        self.relevant_sessions = set()
        self.last_user_items = {}  # user_based # to extend the session model
        self.recent_user_items = {}  # user_based # to remind
        self.recent_user_sessions = {}  # user_based # to remind
        self.user_item_intensity = dict()  # user_based # to remind (for 'session_similarity')

        # cache relations once at startup (in fit)
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1
        self.session_user_map = dict()  # user_based

        self.sim_time = 0
        self.boost_own_count = 0  # user_based
        self.boost_own_count_all = 0  # user_based

    def fit(self, data, items=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''

        if self.last_n_days != None:
            max_time = dt.fromtimestamp(data[self.time_key].max())
            date_threshold = max_time.date() - td(self.last_n_days)
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[data[self.time_key] >= stamp]

        else:
            train = data

        self.num_items = train[self.item_key].max()

        # get the position of the columns
        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        index_user = train.columns.get_loc(self.user_key)  # user_based

        session = -1
        session_items = set()
        time = -1
        user = -1  # user_based
        # cnt = 0
        prev_s_id = -1
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                    self.session_user_map.update({session: user})  # user_based
                    if time < self.min_time:
                        self.min_time = time
                user = row[index_user]  # user_based
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

            # add last viewed items (by the user) to the last_user_items dictionary
            if self.extend_session_length is not None:  # user_based
                if not row[index_user] in self.last_user_items:
                    self.last_user_items[row[index_user]] = []  # create a new list to save the user's last viewed items
                self.last_user_items[row[index_user]].append(row[index_item])
                if len(self.last_user_items[row[index_user]]) > self.extend_session_length:
                    self.last_user_items[row[index_user]] = self.last_user_items[row[index_user]][
                                                            -self.extend_session_length:]

            # reminders
            if self.reminders:  # user_based  # for 'session_similarity' or 'recency'
                if not row[index_user] in self.recent_user_sessions:
                    self.recent_user_sessions[
                        row[index_user]] = []  # create a new set to save the user's last sessions' id
                    self.recent_user_items[row[index_user]] = []
                if row[index_session] != prev_s_id:  # Just once the new session starts
                    self.recent_user_sessions[row[index_user]].append(row[index_session])
                    self.session_items_dict = dict()
                    self.session_items_list = []
                    self.session_items_list.append(row[index_item])
                    self.session_items_dict[row[index_session]] = self.session_items_list
                    self.recent_user_items[row[index_user]].append(self.session_items_dict)
                    # just keep last N sessions
                    if len(self.recent_user_sessions[row[index_user]]) > self.remind_sessions_num:
                        session_id_key = self.recent_user_sessions[row[index_user]][0]
                        del self.recent_user_sessions[row[index_user]][0]  # delete first session in the list
                        del self.recent_user_items[row[index_user]][0][session_id_key]
                        del self.recent_user_items[row[index_user]][0]  # remove first session in the list
                    # do not need to add this session id again!
                    prev_s_id = row[index_session]

                else:
                    self.session_items_list.append(row[index_item])

        # save item_intensity in the last N session for each user
        if self.reminders:  # user_based

            if self.remind_strategy == 'session_similarity':
                for u_id in self.recent_user_sessions:  # todo: if save all users session, also can calculate it for all sessions (not just for tha last N sessions)
                    item_intensity_series = pd.Series()
                    for session_item_dic in self.recent_user_items[u_id]:
                        for s_id, i_list in session_item_dic.items():
                            for i_id in i_list:
                                if not i_id in item_intensity_series.index:  # first occurrence of the item for the user
                                    item_intensity_series.loc[i_id] = 1
                                    # item_intensity_series.set_value(i_id, 1)
                                else:
                                    new_count = item_intensity_series.loc[
                                                    i_id] + 1  # increase the number of occurrence of the item for the user
                                    item_intensity_series.loc[i_id] = new_count

                    item_intensity_series.sort_values(ascending=False, inplace=True)
                    self.user_item_intensity[u_id] = item_intensity_series

            if self.remind_strategy == 'recency':
                # for each item, get the timestamp of last interaction with it
                self.user_item_recency = train.groupby([self.user_key, self.item_key], as_index=False).last()
                # just keep columns UserId, ItemId and Time
                self.user_item_recency = self.user_item_recency[[self.user_key, self.item_key, self.time_key]]
                # sort items by their UserId and Time
                self.user_item_recency.sort_values([self.user_key, self.time_key], ascending=[True, False],
                                                   inplace=True)
                self.user_item_recency = self.user_item_recency.set_index([self.user_key])

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})
        self.session_user_map.update({session: user})  # user_based

        if self.idf_weighting or self.idf_weighting_session:
            self.idf = pd.DataFrame()
            self.idf['idf'] = train.groupby(self.item_key).size()
            self.idf['idf'] = np.log(train[self.session_key].nunique() / self.idf['idf'])
            self.idf = self.idf['idf'].to_dict()

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False,
                     mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session

            if (self.extend):  # add evaluated sessions to the maps.
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})
                self.session_user_map.update({self.session: input_user_id})  # user_based

            self.last_ts = -1
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()
            self.items_previous = []  # user_based

        if mode_type == 'view':
            self.session_items.append(input_item_id)
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append(timestamp - self.last_ts)
                self.last_ts = timestamp

        if skip:
            return

        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        # we add extra items form the user profile as long as the session is not long enough!
        if self.extend_session_length is not None and len(
                items) < self.extend_session_length and input_user_id in self.last_user_items:  # user_based
            # update the session with items from the users past
            if len(self.last_user_items[input_user_id]) < self.extend_session_length:
                addItems = len(self.last_user_items[input_user_id]) - len(self.session_items) + 1
                prev_items = self.last_user_items[input_user_id][:addItems]
            else:  # there are enough items in the use's history
                addItems = self.extend_session_length - len(self.session_items) + 1  # it is always positive
                prev_items = self.last_user_items[input_user_id][1:addItems]
            items = prev_items + self.session_items
            self.extend_session = True  # need to find also relevant sessions for added items

        elif self.extend_session:  # the session length become long enough, then we refine the self.relevant_sessions which just consider current session's items
            self.relevant_sessions = set()
            for item in set(self.session_items):
                self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
                    item)  # then we can continue with just adding related sessions to the current item

            self.extend_session = False  # no added items, so do not need to find relevant sessions for "added items" ALSO no need to refine the self.relevant_sessions (it has been done once)

        # user_based --- start
        if len(self.items_previous) > 0:  # if it is not the first time we are going to predict in this session
            for i in range(len(self.items_previous)):
                if not self.items_previous[
                           i] in items:  # if there is any items that was in the previous step, but not in the current step anymore, we need to refine the self.relevant_sessions
                    self.relevant_sessions = set()
                    for item in set(items):
                        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(item)
                    break  # refined!

        self.items_previous = items
        # user_based --- end

        neighbors = self.find_neighbors(items, input_item_id, session_id, self.dwelling_times, timestamp,
                                        input_user_id)  # user_based

        scores = self.score_items(neighbors, items, timestamp)

        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        predict_for_items = predict_for_item_ids[mask]
        values = [scores[x] for x in predict_for_items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.reminders:  # user_based
            reminder_series = pd.Series()

            if self.remind_strategy == 'session_similarity':  # score = score (Intensity) * 1 [if in top N sessions]
                past_user_sessions = self.calc_similarity(items, self.recent_user_sessions[input_user_id],
                                                          self.dwelling_times, timestamp, input_user_id)
                past_user_sessions = sorted(past_user_sessions, reverse=True, key=lambda x: x[1])
                for sessions_sim_tuple in past_user_sessions:
                    s_id = sessions_sim_tuple[0]
                    for i_id in self.items_for_session(s_id):
                        if not i_id in reminder_series.index:
                            intensity = self.user_item_intensity[input_user_id].loc[i_id]
                            reminder_series.loc[i_id] = intensity

                reminder_series.sort_values(ascending=False, inplace=True)  # (series) index: item_id , value: intensity
                reminder_series = reminder_series.astype(float)  # convert data type from int to float

            if self.remind_strategy == 'recency':  # score = max( time(t) )
                reminder_series = self.user_item_recency.loc[input_user_id]
                if not isinstance(reminder_series, pd.DataFrame):
                    reminder_series = pd.DataFrame({self.item_key: [reminder_series[self.item_key].astype(int)],
                                                    self.time_key: [reminder_series[self.time_key]]}
                                                   , columns=[self.item_key, self.time_key])

                reminder_series = reminder_series.set_index(
                    [self.item_key])  # (dataframe) index: item_id, columns: item_id, time
                reminder_series = reminder_series.iloc[:, 0]  # convert DataFrame to Series
                reminder_series = reminder_series.astype(float)  # convert data type from int to float

            if len(reminder_series) > 0:

                # sort the predictions (sort recommendable items according to their scores)
                reminder_series = reminder_series.iloc[:self.reminders_num]
                k = self.reminders_num
                if len(reminder_series) < k:
                    k = len(reminder_series)

                if k > 0:  # there are any items to remind

                    series.sort_values(ascending=False, inplace=True)
                    # series = series.iloc[:20]  # just keep the first 20 items in the sorted recommendation list

                    if self.remind_mode == 'top':

                        for idx in reminder_series.index:  # check if reminder items are already in the recommendation list or not
                            if idx in series[:20].index:
                                series = series.drop(idx)  # becuase it will be added in the top of the list

                        series = series.iloc[
                                 :(20 - k)]  # just keep the first (20-k) items in the sorted recommendation list
                        reminder_series = reminder_series.iloc[:k]  # keep the first k items in the sorted list

                        base_score = series.iloc[0]
                        for index, value in reminder_series.items():
                            reminder_series[index] = base_score + (k * 0.01)
                            k = k - 1

                    elif self.remind_mode == 'end':

                        for idx in reminder_series.index:  # check if reminder items are already in the recommendation list or not
                            if idx in series[:20].index:
                                reminder_series = reminder_series.drop(
                                    idx)  # because it is already in the recommendation list
                                k = k - 1

                        series = series.iloc[
                                 :(20 - k)]  # just keep the first (20-k) items in the sorted recommendation list
                        reminder_series = reminder_series.iloc[:k]  # keep the first k items

                        base_score = series.iloc[19 - k]
                        k = 1
                        for index, value in reminder_series.items():
                            reminder_series[index] = base_score - (k * 0.01)
                            k = k + 1

                    series = series.append(reminder_series)

        if self.push_reminders:  # give more score to the items that belongs to the current session

            if self.extend_session:
                session_items_series = pd.Series(items)
            else:
                session_items_series = pd.Series(self.session_items)

            session_count = session_items_series.groupby(session_items_series).count() + 1
            # multiply the score of the item in the current session by its number of time in the current session
            series[session_count.index] *= session_count  # TODO: contains the same item several times

        if self.add_reminders:  # force the last 3 items of the current session to be in the top 20 recommendable items
            if self.extend_session:
                session_series = pd.Series(index=items, data=series[
                    items])  # scores of the items that are belong to the current session
            else:
                session_series = pd.Series(index=self.session_items, data=series[
                    self.session_items])  # scores of the items that are belong to the current session
            session_series = session_series[session_series > 0]  # keep them if their scores > 0
            if len(session_series) > 0:
                session_series = session_series.iloc[:3]  # keep the first 3 items
                # sort the predictions (sort recommendable items according to their scors)
                series.sort_values(ascending=False, inplace=True)
                session_series = session_series[session_series < series.iloc[19 - 3]]  # TODO: 19-3
                series[session_series.index] = series.iloc[19 - 3] + 1e-4  # TODO: 1e-4

        # TODO: check if we need update to update the self.last_user_items ?! (not for user_based data splitting, but what about time_based one? )
        # add user to the known-users [last_user_items]
        # if not input_user_id in self.last_user_items:  # user_based
        #     self.last_user_items[input_user_id] = []
        # self.last_user_items[input_user_id].append( input_item_id ) # add current item id to the last_user_items
        # if self.extend_session_length is not None:
        #     if len(self.last_user_items[input_user_id]) > self.extend_session_length: # to keep the length of the last veiwed items as large as self.extend_session_length
        #         self.last_user_items[input_user_id] = self.last_user_items[input_user_id][-self.extend_session_length:]

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)

        Parameters
        --------
        sessions: set

        Returns
        --------
        out : dict
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]

        result = sum / len(map)

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_item_map.get(session);

    def vec_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_vec_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item

        Parameters
        --------
        item: Id of the item session

        Returns
        --------
        out : set
        '''
        return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id, user_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''

        # if extending:  # user_based
        if self.extend_session:  # need to also search for sessions shared the added items
            for item in set(session_items):
                self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(item)
            # self.extend_session = False # we need to search for sessions shared the added items once at the beginning of the session
        else:
            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
                input_item_id)  # add relevant sessions for the current item

        if self.past_neighbors:  # user-based
            for neighbor_sid in self.relevant_sessions:
                if self.session_user_map[neighbor_sid] == user_id:
                    for item in self.items_for_session(neighbor_sid):
                        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(item)

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            possible_neighbors = self.relevant_sessions

        else:  # sample some sessions
            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                possible_neighbors = sample
            else:
                possible_neighbors = self.relevant_sessions

        return possible_neighbors

    def calc_similarity(self, session_items, sessions, dwelling_times, timestamp, user_id):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        pos_map = {}
        length = len(session_items)

        count = 1
        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1

        if self.dwelling_time:
            dt = dwelling_times.copy()
            dt.append(0)
            dt = pd.Series(dt, index=session_items)
            dt = dt / dt.max()
            # dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
            dt[session_items[-1]] = 1

            # print(dt)
            for i in range(len(dt)):
                pos_map[session_items[i]] *= dt.iloc[i]
            # print(pos_map)

        if self.idf_weighting_session:
            max = -1
            for item in session_items:
                pos_map[item] = self.idf[item] if item in self.idf else 0
        #                 if pos_map[item] > max:
        #                     max = pos_map[item]
        #             for item in session_items:
        #                 pos_map[item] = pos_map[item] / max

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            n_items = self.items_for_session(session)
            sts = self.session_time[session]

            # dot product
            similarity = self.vec(items, n_items, pos_map)
            if similarity > 0:

                if self.weighting_time:
                    diff = timestamp - sts
                    days = round(diff / 60 / 60 / 24)
                    decay = pow(7 / 8, days)
                    similarity *= decay

                # print("days:",days," => ",decay)

                if self.boost_own_sessions is not None and self.boost_own_sessions > 0.0 and self.session_user_map[
                    session] == user_id:  # user_based
                    similarity = similarity + (similarity * self.boost_own_sessions)
                    self.boost_own_count += 1

                self.boost_own_count_all += 1  # user_based

                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id, dwelling_times, timestamp, user_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id.

        Parameters
        --------
        session_items: set of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id,
                                                             user_id)  # user_based
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors, dwelling_times, timestamp,
                                                  user_id)  # user_based

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        item_set = set(current_session)
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            step = 1

            for item in reversed(current_session):
                if item in items:
                    decay = getattr(self, self.weighting_score + '_score')(step)
                    break
                step += 1

            for item in items:

                if not self.remind and item in item_set:  # d=0* (exclude items form the current session)
                    continue  # dont want to remind user the item that is already is his current (extended) session

                old_score = scores.get(item)
                new_score = session[1]
                new_score = new_score if not self.idf_weighting else new_score + (
                            new_score * self.idf[item] * self.idf_weighting)
                new_score = new_score * decay

                if not old_score is None:
                    new_score = old_score + new_score

                scores.update({item: new_score})

        return scores

    def linear_score(self, i):
        return 1 - (0.1 * i) if i <= 100 else 0

    def same_score(self, i):
        return 1

    def div_score(self, i):
        return 1 / i

    def log_score(self, i):
        return 1 / (log10(i + 1.7))

    def quadratic_score(self, i):
        return 1 / (i * i)

    def linear(self, i, length):
        return 1 - (0.1 * (length - i)) if i <= 10 else 0

    def same(self, i, length):
        return 1

    def div(self, i, length):
        return i / length

    def log(self, i, length):
        return 1 / (log10((length - i) + 1.7))

    def quadratic(self, i, length):
        return (i / length) ** 2

    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.session_user_map = dict()  # user_based

    def support_users(self):
        '''
            whether it is a session-based or session-aware algorithm
            (if returns True, method "predict_with_training_data" must be defined as well)

            Parameters
            --------

            Returns
            --------
            True : if it is session-aware
            False : if it is session-based
        '''
        return True

    def predict_with_training_data(self):
        '''
            (this method must be defined if "support_users is True")
            whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

            Parameters
            --------

            Returns
            --------
            True : e.g. hgru4rec
            False : e.g. uvsknn
            '''
        return False