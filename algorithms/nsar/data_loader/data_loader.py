import numpy as np
from datetime import datetime

class DataLoader(object):
    def __init__(self, config):
        self._max_length = config['max_length']
        self._batch_size = config['batch_size']
        self._batch_index = -1
        self._data = None
        self._num_events = None
        self._num_events_eval = None
        self._num_batch = None

    def load_data(self, data, user2id, item2id):
        self._data = []
        self._num_events = 0
        self.sid_to_index = dict()
        for sess_id in data.SessionId.unique():
            session_pd = data.loc[data['SessionId'] == sess_id]

            session = [[user2id[int(row[0])], item2id[int(row[1])]] + list(self.extract_time_context_utc(row[2])) for row in
                       session_pd[['UserId', 'ItemId', 'Time']].values ]

            self._num_events += len(session)

            if len(session) < self._max_length + 1:
                for _ in range(self._max_length - len(session) + 1):
                    session.append([0] * 5)


            self.sid_to_index[sess_id] = [len(self._data)]

            self._data.append(session)

        self._data = np.array(self._data, dtype=np.int32)
        self._num_events_eval = self._num_events - len(self._data)
        self._num_batch = int(float(len(self._data) - 1) / self._batch_size) + 1

        print('--- Data ---')
        print('Num sessions: ', len(self._data))
        print('Num events: ', self._num_events)

    def next_epoch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self._data)
        self._batch_index = 0

    def next_batch(self):
        start_idx = self._batch_index * self._batch_size
        end_idx = start_idx + self._batch_size
        self._batch_index += 1
        if self._batch_index == self._num_batch:
            self._batch_index = -1
        return self._data[start_idx: end_idx]

    def has_next(self):
        return self._batch_index != -1

    def extract_time_context_utc(self, utc):

        dt = datetime.utcfromtimestamp(float(utc))
        hour = dt.hour
        month = dt.month
        week_day = dt.weekday()
        if month == 12:
            day_of_month = datetime(day=1, month=1, year=dt.year + 1) \
                           - datetime(day=1, month=month, year=dt.year)
        else:
            day_of_month = datetime(day=1, month=month + 1, year=dt.year) \
                           - datetime(day=1, month=month, year=dt.year)
        day_of_month = day_of_month.days
        if dt.day < day_of_month / 2:
            half_month_ped = month * 2 - 1
        else:
            half_month_ped = month * 2
        return hour, week_day, half_month_ped

    def data_from_sid(self, sid):

        return self._data[self.sid_to_index[sid]]