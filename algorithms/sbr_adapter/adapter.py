import numpy as np
import pandas as pd
import algorithms.sbr_adapter.factorization.fpmc as fpmc
import algorithms.sbr_adapter.factorization.fossil as fossil
import algorithms.sbr_adapter.factorization.fism as fism
import algorithms.sbr_adapter.factorization.bprmf as bprmf


class Adapter:
    '''
    Adapter(algo='fpmc', params={}, session_key='ItemId', item_key='ItemId')

    Popularity predictor that gives higher scores to items with larger support.

    The score is given by:

    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}

    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    support_by_key : string or None
        If not None, count the number of unique values of the attribute of the training data given by the specified header. If None, count the events. (Default value: None)

    '''

    def __init__(self, algo='fpmc', params={}, session_key='SessionId', item_key='ItemId'):
        self.algo = algo
        self.item_key = item_key
        self.session_key = session_key

        if self.algo == 'fpmc':
            self.instance = fpmc.FPMC()
        elif self.algo == 'fossil':
            self.instance = fossil.Fossil()
        elif self.algo == 'fism':
            self.instance = fism.FISM()
        elif self.algo == 'bprmf':
            self.instance = bprmf.BPRMF()

        self.current_session = None

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        self.instance.prepare_model(data)
        self.instance.train(data)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        iidx = self.instance.item_map[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)

        out = self.instance.recommendations([[iidx]], session=self.session)

        return pd.Series(data=out, index=self.instance.item_list)
