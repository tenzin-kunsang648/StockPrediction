import stockLSTM
import pandas as pd


def findBestNumNodes(data, hiddenNodesList):
    '''This function performs cross-validation on varying number of hidden nodes,
    holding other parameters constant.

    data = cross-validation data
    hiddenNodesList = list of number of hidden nodes'''

    accuracyList = []

    for numNodes in hiddenNodesList:
        accuracy = stockLSTM.forwardChainingCV(data, 60, numNodes, 4)
        accuracyList.append(accuracy)

    return accuracyList, accuracyList.index(min(accuracyList))


def findBestNumLayers(data, hiddenLayersList):
    '''This function performs cross-validation on varying number of hidden layers, 
    holding other parameters constant.

    data = cross-validation data
    hiddenLayersList = list of number of hidden layers'''

    accuracyList = []

    for numLayers in hiddenLayersList:
        accuracy = stockLSTM.forwardChainingCV(data, 60, 100, numLayers)
        accuracyList.append(accuracy)

    return accuracyList, accuracyList.index(min(accuracyList))


def findBestCombo(data, hiddenNodesList, hiddenLayersList):
    '''This function performs cross-validation on combinations of number of 
    hidden layers and hidden nodes, holding other parameters constant.

    data = cross-validation data
    hiddenNodesList = list of number of hidden nodes
    hiddenLayersList = list of number of hidden layers'''

    accuracyList = []

    for numNodes in hiddenNodesList:
        for numLayers in hiddenLayersList:
            accuracy = stockLSTM.forwardChainingCV(
                data, 60, numNodes, numLayers)
            accuracyList.append(accuracy)

    return accuracyList, accuracyList.index(min(accuracyList))


def main():
    '''Main function takes the cross-validation data and implements three
    cross-validation processes to find the best model for stock prediction. 
    It sends lists of variations of number of hidden layers, hidden nodes, 
    and combination of hidden nodes and hidden layers.'''

    data = pd.read_csv('AAPL_CV.csv')

    accuracyListNodes, minMAEIndexNodes = findBestNumNodes(
        data, [20, 40, 60, 80, 100])
    print(accuracyListNodes, minMAEIndexNodes,
          accuracyListNodes[minMAEIndexNodes])
    accuracyListLayers, minMAEIndexLayers = findBestNumLayers(data, [
                                                              1, 2, 3, 4, 5])
    print(accuracyListLayers, minMAEIndexLayers,
          accuracyListLayers[minMAEIndexLayers])
    accuracyListCombo, minMAEIndexCombo = findBestCombo(
        data, [20, 40, 60, 80, 100], [1, 2, 3, 4, 5])
    print(accuracyListCombo, minMAEIndexCombo,
          accuracyListCombo[minMAEIndexCombo])


if __name__ == "__main__":
    main()
