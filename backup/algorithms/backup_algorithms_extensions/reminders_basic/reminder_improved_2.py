import pandas as pd


class Reminder:
    '''

    reminders: bool
        Include reminding items in the (main) recommendation list. (default: False)

    remind_strategy: string
        Ranking strategy of the reminding list (default: recency)
        'recency', 'session_similarity', 'hybrid' (only in improved version)

    remind_sessions_num: int
        Number of the last user's sessions that the possible items for reminding are taken from (default: 6)

    reminders_num: int
        length of the reminding list (default: 3)

    remind_mode: string
        The postion of the remining items in recommendation list (top, end). (default: end)


    '''

    def __init__(self, remind_strategy='recency', remind_sessions_num=6, reminders_num=3, remind_mode='end', cut_off_threshold=20, improved_version=False, weight_base=1, weight_IRec=0, weight_SSim=0):
        self.remind_sessions_num = remind_sessions_num
        self.remind_strategy = remind_strategy
        self.recent_user_items = {}  # user_based # to remind
        self.recent_user_sessions = {}  # user_based # to remind
        self.prev_session_id = -1
        self.improved_version = improved_version
        if self.improved_version:
            self.w_base = weight_base
            self.w_IRec = weight_IRec
            self.w_SSim = weight_SSim
        else:
            self.remind_mode = remind_mode
            self.reminders_num = reminders_num
            self.cut_off = cut_off_threshold
            if self.remind_strategy == 'session_similarity':
                self.user_item_intensity = dict()  # user_based # to remind (for 'session_similarity')

    def reminders_fit_in_loop(self, row, index_user, index_session, index_item):
        if not row[index_user] in self.recent_user_sessions:
            # create a new set to save the user's last sessions' id
            self.recent_user_sessions[row[index_user]] = []
            self.recent_user_items[row[index_user]] = []
        if row[index_session] != self.prev_session_id:  # Just once the new session starts
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
            self.prev_session_id = row[index_session]

        else:
            self.session_items_list.append(row[index_item])
        pass

    def reminders_fit(self, train, user_key, item_key, time_key):
        if self.remind_strategy == 'recency' or self.remind_strategy == 'hybrid':
            # for each item, get the timestamp of last interaction with it
            self.user_item_recency = train.groupby([user_key, item_key], as_index=False).last()
            # just keep columns UserId, ItemId and Time
            self.user_item_recency = self.user_item_recency[[user_key, item_key, time_key]]
            # sort items by their UserId and Time
            self.user_item_recency.sort_values([user_key, time_key], ascending=[True, False],
                                               inplace=True)
            # self.user_item_recency is a DataFrame with index=UserId, and 2 columns: ItemId & Time (sorted by UserId & Time)
            self.user_item_recency = self.user_item_recency.set_index([user_key])

        elif self.remind_strategy == 'session_similarity' and not self.improved_version:
            for u_id in self.recent_user_sessions:
                item_intensity_series = pd.Series()
                for session_item_dic in self.recent_user_items[u_id]:
                    for s_id, i_list in session_item_dic.items():
                        for i_id in i_list:
                            if not i_id in item_intensity_series.index:  # first occurrence of the item for the user
                                item_intensity_series.loc[i_id] = 1
                                # item_intensity_series.set_value(i_id, 1)
                            else:
                                # increase the number of occurrence of the item for the user
                                new_count = item_intensity_series.loc[i_id] + 1
                                item_intensity_series.loc[i_id] = new_count

                item_intensity_series.sort_values(ascending=False, inplace=True)
                self.user_item_intensity[u_id] = item_intensity_series

        pass

    def reminders_predict_next(self, input_user_id, series, item_key, time_key, past_user_sessions = None, session_item_map = None, input_timestamp = None):
        # create reminder list(s)
        reminder_series = pd.Series()
        if self.improved_version:
            reminder_series_IRec = pd.Series()
            reminder_series_SSim = pd.Series()

        if self.remind_strategy == 'hybrid' and self.w_IRec != 0:
            # create a series with same index as the recommendation list but with values 0
            reminder_series = series.copy()
            reminder_series.loc[:] = 0

            user_item_recency_series = self.user_item_recency.loc[input_user_id]
            if not isinstance(user_item_recency_series, pd.DataFrame):  # convert (type of) Series to DataFrame
                user_item_recency_series = pd.DataFrame({item_key: [user_item_recency_series[item_key].astype(int)],
                                                         time_key: [user_item_recency_series[time_key]]}
                                                        , columns=[item_key, time_key])

            user_item_recency_series = user_item_recency_series.set_index(
                [item_key])  # (dataframe) index: item_id, columns: item_id, time
            user_item_recency_series = user_item_recency_series.iloc[:, 0]  # convert DataFrame to Series
            user_item_recency_series = user_item_recency_series.astype(float)  # convert data type from int to float

            for index, value in user_item_recency_series.items():
                reminder_series.loc[index] = input_timestamp / (input_timestamp - value)
            reminder_series_IRec = reminder_series.copy()

        if self.remind_strategy == 'hybrid' and self.w_SSim != 0:  # score = score (Intensity) * 1 [if in top N sessions]
            past_user_sessions = sorted(past_user_sessions, reverse=True, key=lambda x: x[1])
            # create a series with same index as the recommendation list but with values 0
            reminder_series = series.copy()
            reminder_series.loc[:] = 0

            for sessions_sim_tuple in past_user_sessions:
                s_id = sessions_sim_tuple[0]
                s_score = sessions_sim_tuple[1]
                sessions_items = session_item_map.get(s_id)
                for i_id in sessions_items:
                    # if the item has not been added to the reminder list yet
                    if not i_id in reminder_series.index:
                        reminder_series.loc[i_id] = s_score
                    else:
                        reminder_series.loc[i_id] = reminder_series.loc[i_id] + s_score  # cumulative sum

            reminder_series.sort_values(ascending=False, inplace=True)
            # reminder_series = reminder_series.astype(float)  # convert data type from int to float
            reminder_series_SSim = reminder_series.copy()

        if self.remind_strategy == 'recency':  # score = max( time(t) )
            reminder_series = self.user_item_recency.loc[input_user_id]
            if not isinstance(reminder_series, pd.DataFrame):  # convert (type of) Series to DataFrame
                reminder_series = pd.DataFrame({item_key: [reminder_series[item_key].astype(int)],
                                                time_key: [reminder_series[time_key]]}
                                               , columns=[item_key, time_key])

            reminder_series = reminder_series.set_index([item_key])  # (dataframe) index: item_id, columns: item_id, time
            reminder_series = reminder_series.iloc[:, 0]  # convert DataFrame to Series
            reminder_series = reminder_series.astype(float)  # convert data type from int to float

        if self.remind_strategy == 'session_similarity':  # score = score (Intensity) * 1 [if in top N sessions]
            past_user_sessions = sorted(past_user_sessions, reverse=True, key=lambda x: x[1])
            for sessions_sim_tuple in past_user_sessions:
                s_id = sessions_sim_tuple[0]
                sessions_items = session_item_map.get(s_id)
                for i_id in sessions_items:
                    # if the item has not been added to the reminder list yet
                    if not i_id in reminder_series.index:
                        intensity = self.user_item_intensity[input_user_id].loc[i_id]
                        reminder_series.loc[i_id] = intensity

            reminder_series.sort_values(ascending=False, inplace=True)  # (series) index: item_id , value: intensity
            reminder_series = reminder_series.astype(float)  # convert data type from int to float

        # combine reminder list(s) with the recommendation list
        if self.improved_version:

            # normalization
            if series.max() != 0:
                series_base = series.copy() / series.max()
            if self.w_IRec != 0 and reminder_series_IRec.max() != 0:
                reminder_series_IRec = reminder_series_IRec / reminder_series_IRec.max()
            if self.w_SSim != 0 and reminder_series_SSim.max() != 0:
                reminder_series_SSim = reminder_series_SSim / reminder_series_SSim.max()

            # weighted sum to combine the scores
            # series = (self.w_base * series_base) + (self.w_IRec * reminder_series_IRec) + (self.w_SSim * reminder_series_SSim)
            series = self.w_base * series_base
            if self.w_IRec != 0:
                series = series + (self.w_IRec * reminder_series_IRec)
            if self.w_SSim != 0:
                series = series + (self.w_SSim * reminder_series_SSim)

        # basic reminders
        elif len(reminder_series) > 0:

            # sort the predictions (sort recommendable items according to their scores)
            reminder_series = reminder_series.iloc[:self.reminders_num]
            k = self.reminders_num
            if len(reminder_series) < k:
                k = len(reminder_series)

            if k > 0:  # there are any items to remind

                # series.sort_values(ascending=False, inplace=True)
                series = series.sort_values(ascending=False).copy()
                # series = series.iloc[:20]  # just keep the first 20 items in the sorted recommendation list

                if self.remind_mode == 'top':

                    for idx in reminder_series.index:
                        # check if reminder items are already in the recommendation list or not
                        if idx in series[:self.cut_off].index:
                            # because it will be added in the top of the list
                            series = series.drop(idx)

                    # just keep the first (20-k) items in the sorted recommendation list
                    series = series.iloc[:(self.cut_off - k)]
                    # keep the first k items in the sorted list
                    reminder_series = reminder_series.iloc[:k]

                    base_score = series.iloc[0]
                    for index, value in reminder_series.items():
                        reminder_series[index] = base_score + (k * 0.01)
                        k = k - 1

                    series = series.append(reminder_series)

                elif self.remind_mode == 'end':

                    for idx in reminder_series.index:
                        # check if reminder items are already in the recommendation list or not
                        if idx in series[:self.cut_off].index:
                            # because it is already in the recommendation list
                            reminder_series = reminder_series.drop(idx)
                            k = k - 1

                    # just keep the first (20-k) items in the sorted recommendation list
                    series = series.iloc[:(self.cut_off - k)]
                    # keep the first k items
                    reminder_series = reminder_series.iloc[:k]

                    base_score = series.iloc[(self.cut_off - 1) - k]
                    k = 1
                    for index, value in reminder_series.items():
                        reminder_series[index] = base_score - (k * 0.01)
                        k = k + 1

                    series = series.append(reminder_series)

        return series


