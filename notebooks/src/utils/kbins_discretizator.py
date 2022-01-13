from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


class KBinsDiscretizator():
    def __init__(self, lower_bounds, upper_bounds, bins_array, encode='ordinal', strategy='uniform'):
        self.bins_array = bins_array
        self.discretizer = KBinsDiscretizer(
            n_bins=bins_array, encode=encode, strategy=strategy)
        self.discretizer.fit([lower_bounds, upper_bounds])

    def idx_state(self, state):
        return np.ravel_multi_index((self.discretizer.transform([state])[0]).astype(int), self.bins_array)

    def get_n_states(self):
        return np.prod(self.bins_array)
