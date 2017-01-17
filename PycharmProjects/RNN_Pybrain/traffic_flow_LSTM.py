# Import statements required to build, train, and test the networks
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.datasets import mnist
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import ModuleValidator

# Import statement to keep track of time elapsed
import time
# Import statement to plot data
import matplotlib.pyplot as plt

path = "MNIST/"
(train, test) = mnist.makeMnistDataSets(path)

# Build the networks with increasing complexity in number of nodes in order to test if number of nodes affects error rate
# Trainers are created for each network and each network is trained once
# Then, I test the network against the actual dataset and get the mean squared error and see the connection between nodes and MSE
inputNode = 28*28
baselineNumberOfNodes = 10

numOutputNodes = 10
LSTMNetworkNumber = 1

LSTMNetworks = []
multipliers = [1,2,5,10]
overallError = []
testComparison = []
timeSpentBuilding = []
timeSpentTraining = []
timeSpentTesting = []
for i in range (0, len(multipliers)):
    print("LSTMNetwork" + str(LSTMNetworkNumber) + " created")
    start = time.clock()
    LSTMNetwork = buildNetwork(inputNode, multipliers[i] * baselineNumberOfNodes, numOutputNodes, hiddenclass = LSTMLayer, recurrent = True)
    end = time.clock()
    print str(end - start) + " seconds for the creation of the network"
    LSTMNetworks.append(LSTMNetwork)
    timeSpentBuilding.append(end - start)

    print("Trainer created")
    start = time.clock()
    trainer = BackpropTrainer(LSTMNetwork, dataset=train)
    error = trainer.train()
    end = time.clock()
    print str(end - start) + " seconds for the training the network"
    print("LSTMNetwork" + str(LSTMNetworkNumber) + " has an error rate of: " + str(error))
    overallError.append(error)
    timeSpentTraining.append(end - start)

    print("Validating the trained network to the testing one")
    start = time.clock()
    validation = ModuleValidator.MSE(LSTMNetwork, test)
    end = time.clock()
    print str(end - start) + " seconds for testing the network"
    print("Backpropagation Trainer " + str(LSTMNetworkNumber) + " has test error rate of: " + str(validation))
    testComparison.append(validation)
    timeSpentTesting.append(end - start)

    LSTMNetworkNumber += 1

# Values for OverallError and TestComparison
print str(overallError)
print str(testComparison)

# Plot of network's error after training
plt.plot([1, 2, 3, 4], overallError)
plt.xlabel("LSTM Network w/ 1, 2, 5, 10 * " + str(baselineNumberOfNodes) + " layers")
plt.ylabel("Error After Training")
plt.show()

# Plot of network's test error rate matched to actual dataset in terms of mean squared error
plt.plot([1, 2, 3, 4], testComparison)
plt.xlabel("LSTM Network w/ 1, 2, 5, 10 * " + str(baselineNumberOfNodes) + " layers")
plt.ylabel("MSE of Testing")
plt.show()

# Maybe plot the time it takes to either train and or test the networks?
plt.plot([1, 2, 3, 4], timeSpentBuilding, "bo")
plt.xlabel("LSTM Network w/ 1, 2, 5, 10 * " + str(baselineNumberOfNodes) + " layers")
plt.ylabel("Time spent building in seconds")
plt.show()

plt.plot([1, 2, 3, 4], timeSpentTraining, "bo")
plt.xlabel("LSTM Network w/ 1, 2, 5, 10 * " + str(baselineNumberOfNodes) + " layers")
plt.ylabel("Time spent training in seconds")
plt.show()

plt.plot([1, 2, 3, 4], timeSpentTesting, "bo")
plt.xlabel("LSTM Network w/ 1, 2, 5, 10 * " + str(baselineNumberOfNodes) + " layers")
plt.ylabel("Time spent testing in seconds")
plt.show()