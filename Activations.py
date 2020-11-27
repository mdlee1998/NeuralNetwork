import numpy as np

class ReLU:

    def forwardProp(self, input):
        self.output = np.maximum(0, input)
