import numpy as np
import math


def transferFunction(x, Derivative = False):

    if Derivative == False:
        return np.tanh(x)
    else:
        return 1.0-np.tan(x)**2

def transferFunctionOut(x, Derivative = False):

    if Derivative == False:
        return 1/(1+np.exp(-x))
    else:
        return transferFunction(x)*(1-transferFunction(x))

class Net(object):
    """class for constructing neural net"""

    def __init__(self, topology):
        """initialization of weights"""
        self.weights = []
        self.deltaWeights = []
        self.gradients = []
        self.OutMatrix =[]
        self.totalNetError = 0
        self.currentNetError = 0
        self.Error=0

        for i in range(1,len(topology)-1):
            self.weights.append(np.random.randn(topology[i-1]+1, topology[i]).astype(np.float32))
        self.weights.append(np.random.randn(topology[i] + 1, topology[i+1]).astype(np.float32))

        for k in range(len(self.weights)):
            self.deltaWeights.append(np.zeros(self.weights[k].shape,dtype=np.float32))



    def feedForward(self,inputVals, Bias=1):
        """train the network"""
        inputVals = np.append(inputVals,Bias).astype(np.float32)
        self.OutMatrix = [inputVals]

        for j in range(len(self.weights)-1):
            self.OutMatrix.append(transferFunction(np.dot(self.OutMatrix[j], self.weights[j]).astype(np.float32)))
            self.OutMatrix[j+1]=np.append(self.OutMatrix[j+1],Bias)
        self.OutMatrix.append(transferFunctionOut(np.dot(self.OutMatrix[j+1], self.weights[j+1]).astype(np.float32)))

    def getResults(self):
        return self.OutMatrix[-1]

    def backProp(self,targetVals, alpha=0.2,beta = 0.15):
        self.Error = (targetVals - self.OutMatrix[-1])
        self.m_error = 0.5*(self.Error**2)
        self.totalNetError += self.m_error

        #calculate output layer gradients and append it
        self.gradients=[np.array(self.Error*transferFunction(self.OutMatrix[-1], Derivative=True),dtype=np.float32)]


        #calculate hiddenlayer gradients
        for n in range(len(self.OutMatrix)-2,0,-1):

            self.gradients.append(np.multiply(np.dot(self.weights[n][:-1:] ,self.gradients[-1]),
                                              transferFunction(self.OutMatrix[n][:-1:],Derivative=True)))


        self.gradients.reverse()


        #crucial step...update weights.....
        for l in range(len(self.deltaWeights)):
            self.deltaWeights[l] = np.add(np.full(self.deltaWeights[l].T.shape,alpha*self.OutMatrix[l],dtype=np.float32).T*self.gradients[l],beta*self.deltaWeights[l])
            self.weights[l] += self.deltaWeights[l]


    def blah(self):

        print "net error:"
        print  (self.totalNetError)




#if "__name__" == "__main__":
Topology = [2,4,1]
mynet = Net(Topology)
for te in range(0,3000):
    inputval = np.random.randint(2,size=2)
    print "input:" + str(inputval)
    targetval = np.array((inputval[0] ^ inputval[1]))
    print "target: " + str(targetval)
    mynet.feedForward(inputval)
    print "out:" + str(mynet.getResults())
    mynet.backProp(targetval)












