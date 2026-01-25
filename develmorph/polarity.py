import numpy as np


class PolarityField:
    """
    Global polarity field (e.g. anterior-posterior).
    """

    def __init__(self, axis=None):

        if axis is None:
            v = np.random.randn(3)
            self.axis = v / np.linalg.norm(v)
        else:
            self.axis = axis / np.linalg.norm(axis)

    def value(self, pos):
        """
        Return polarity value at position.
        """
        return np.dot(pos, self.axis)
