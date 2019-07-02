class CascadingHybrid:
    '''
    CascadingHybrid(algorithms, filter, mode)

    Parameters
    --------
    algorithms : list
        List of algorithms to combine consecutively.
    filter : list
        List of thresholds or result sizes to define which scores should be retained from the first (n-1) of n algorithms.
        Hence, the list length has to be len(algorithms) - 1.
    mode : string
        Can be 'threshold' or 'rank'.
        Threshold uses a minimal value to filter by score.
        Rank filters the top n results of an algorithm.
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?

    '''

    def __init__(self, algorithms, filter, mode='threshold', fit=True, clearFlag=True):
        self.algorithms = algorithms
        self.filter = filter
        self.mode = mode
        self.run_fit = fit
        self.clearFlag = clearFlag

    def init(self, train, test=None, slice=None):
        for a in self.algorithms: 
            if hasattr(a, 'init'):
                a.init( train, test, slice )

    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        if self.run_fit:
            for a in self.algorithms:
                a.fit(data)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, timestamp=0):
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
        predictions = []
        for a in self.algorithms:
            predictions.append(a.predict_next(session_id, input_item_id, predict_for_item_ids, skip))

        if self.mode == 'threshold':

            mask = []
            i = 0
            while i < len(predictions):
                final = predictions[i]
                final[mask] = 0
                mask = final[final < self.filter[i]].index if len(self.filter) - 1 >= i else None
                i += 1

            return final

        elif self.mode == 'rank':

            mask = []
            i = 0
            while i < len(predictions):
                final = predictions[i]
                final[mask] = 0
                if len(self.filter) - 1 >= i:
                    final.sort_values(ascending=False, inplace=True)
                    mask = final[self.filter[i]:].index
                i += 1

            return final


    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()
