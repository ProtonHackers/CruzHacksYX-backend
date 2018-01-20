import numpy as np

class BackPropNet:
    layer_count = 0
    shape = None
    weights = []

    def __init__(self, layerSize):

        # Layer Information
        self.layer_count = len(layerSize) - 1
        self.shape = layerSize

        # I/O data from previous runs
        self._layerInput = []
        self._layerOutput = []

        # Create the weightage arrays
        # layer1 -> layer1
        for (layer1, layer2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(
                np.random.normal(scale=0.1, size=(layer2, layer1 + 1)))  # can change scale if values too small

    def Run(self, input):

        cases = input.shape[0]

        self._layerInput = []
        self._layerOutput = []

        # Running
        for i in range(self.layer_count):
            # This loop determines layer input
            if i == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, cases])]))  # takes dot product
            else:
                layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1, cases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.Sigmoid(layerInput))

        return self._layerOutput[-1].T

    def TrainEpochs(self, input, target, trainingRate=0.2):

        delta = []
        cases = input.shape[0]

        # run the net
        self.Run(input)

        # then calculate delta
        for i in reversed(range(self.layer_count)):
            if i == self.layer_count - 1:
                # compare to the target
                output_delta = self._layerOutput[i] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.Sigmoid(self._layerInput[i], True))
            else:
                # compare to next layer's delta
                delta_pullback = self.weights[i + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.Sigmoid(self._layerInput[i], True))

        # make weightage
        for i in range(self.layer_count):
            delta_index = self.layer_count - 1 - i

            if i == 0:
                layerOutput = np.vstack([input.T, np.ones([1, cases])])
            else:
                layerOutput = np.vstack([self._layerOutput[i - 1], np.ones([1, self._layerOutput[i - 1].shape[1]])])

            weightDelta = np.sum( \
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0) \
                , axis=0)
            self.weights[i] -= trainingRate * weightDelta

        return error

    def Sigmoid(self, x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.Sigmoid(x)
            return out * (1 - out)

'''Labels and Features'''
'''can set any input, target must be between 0 and 1'''
'''Change BackPropNet shapes accordingly'''

X = np.array([[1, 1, 0, 1], [0, 0, 0, 1], [10, 10, 11, 8]])
y = np.array([[.1, .3], [0, 0.2], [1, 0.7]])

'''Sets input and output shape'''
'''Tells Back Propagation what kind of input and output to expect'''
bpn = BackPropNet((len(X[0]), 2, len(y[0])))
print (bpn.shape) # tuple of input and output shape
print (bpn.weights) # numpy array

# Can change max and lnErr depending on how much accuracy you want
# Larger the max and smaller the lnErr will increase run time
max = 100000
lnErr = 1e-5
for i in range(max + 1):
    err = bpn.TrainEpochs(X, y)
    if i % 5000 == 0:
        print ("Iteration {0} \tError: {1:0.6f}".format(i, err))
    if err <= lnErr:
        print ("Minimum error reached at iteration {0}\n".format(i))
        break

# display output
levelOutput = bpn.Run(X)
print ("Input: {0}\nOutput: {1}".format(X, levelOutput))