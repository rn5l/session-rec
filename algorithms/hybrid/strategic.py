class StrategicHybrid:
    '''
    StrategicHybrid(algorithms, weights)

    Use different algorithms depending on the length of the current session.

    Parameters
    --------
    algorithms : list
        List of algorithms to combine with a switching strategy.
    thresholds : float
        Proper list of session length thresholds.
        For [5,10] the first algorithm is applied until the session exceeds a length of 5 actions, the second up to a length of 10, and the third for the rest.
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?

    '''

    def __init__(self, algorithms, thresholds, fit=True, clearFlag=True):
        self.algorithms = algorithms
        self.thresholds = thresholds
        self.run_fit = fit
        self.clearFlag = clearFlag
    
    def init(self, train, test=None, slice=None):
        for a in self.algorithms: 
            if hasattr(a, 'init'):
                a.init( train, test, slice )
    
    def fit(self, data):
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

        self.session = -1
        self.session_items = []

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

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()

        self.session_items.append(input_item_id)

        predictions = []
        for a in self.algorithms:
            predictions.append(a.predict_next(session_id, input_item_id, predict_for_item_ids, skip))

        for i, prediction in reversed(list(enumerate(predictions))):
            if len(self.thresholds) - 1 < i or len(self.session_items) <= self.thresholds[i]:
                final = prediction

        return final


    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()
