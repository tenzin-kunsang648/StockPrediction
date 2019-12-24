import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics  # for calculating the mean of a list
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Define scaler outside of all functions, so we can use it in every function!
# This scaler transforms a variable to have values between 0 and 1
# by subtracting the min and dividing by max - min
scaler = MinMaxScaler(feature_range=(0, 1))


def createNN(model, layers, numNodes, dropoutPercent):
    '''This function creates the LSTM Neural Network with given
    number of hidden layers, hidden nodes, and dropout percentage.

    model: initialized sequential model
    layers: number of hidden layers
    numNodes: number of hidden nodes in each layer
    dropoutPercent: drop out percentage'''

    for i in range(layers - 1):
        model.add(LSTM(units=numNodes, return_sequences=True))
        model.add(Dropout(dropoutPercent))

    model.add(LSTM(units=numNodes))
    model.add(Dropout(dropoutPercent))
    model.add(Dense(units=1))


def forwardChainingCV(data, numDaysPrevious, numHiddenNodes, numHiddenLayers):
    '''This function performs time series cross-validation (CV) process for 
    a predefined model. It is only called during CV to find our best 
    model for stock prediction.

    data: cross-validation data
    numDaysPrevious: number of previous days used to predict the stock price
    numHiddenNodes: number of hidden nodes in each layer
    numHiddenLayers: number of hidden layers'''

    dataProcessed = data.iloc[:, 1:2].values
    dataScaled = scaler.fit_transform(dataProcessed)
    numTimePeriod = len(data) // numDaysPrevious

    errorList = []

    for i in range(numTimePeriod - 2):
        # Define the range of data in training and testing sets
        start_row_train = i * numDaysPrevious
        end_row_train = (i + 2) * numDaysPrevious
        start_row_test = (i + 2) * numDaysPrevious
        end_row_test = (i + 3) * numDaysPrevious
        trainData = data.iloc[start_row_train:end_row_train, :]
        testData = data.iloc[start_row_test:end_row_test, :]
        testColumn = testData.iloc[:, 1:2].values  # for accuracy test

        # Create an input data for LSTM
        inputsData = []
        labels = []
        for j in range((i + 1) * numDaysPrevious, (i + 2) * numDaysPrevious):
            inputsData.append(dataScaled[j - numDaysPrevious: j, 0])
            labels.append(dataScaled[j, 0])

        inputsData, labels = np.array(inputsData), np.array(labels)
        inputsData = np.reshape(
            inputsData, (inputsData.shape[0], inputsData.shape[1], 1))

        model = Sequential()
        createNN(model, numHiddenLayers, numHiddenNodes, 0.2)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(inputsData, labels, epochs=50, batch_size=32)

        predictions = predictData(trainData, testData, numDaysPrevious, model)
        errorSum = accuracyCal(predictions, testColumn)
        errorList.append(errorSum)
        print("This is sample:", i, " where hiddenNodes is ", str(
            numHiddenNodes) + " and number of layers is ", str(numHiddenLayers))

    avgError = statistics.mean(errorList)

    return avgError


def predictData(trainData, testData, numberOfDays, model):
    '''This function takes in the training and testing data and predicts
    the stock prices of the test data.

    trainData: training data
    testData: testing data
    numberOfDays: number of previous days used to predict the stock price
    model: an LSTM neural network with predefined structure'''

    testColumn = testData.iloc[:, 1:2].values
    totalData = pd.concat((trainData['Open'], testData['Open']), axis=0)

    test_inputs = totalData[len(totalData) -
                            len(testColumn) - numberOfDays:].values
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)

    test_features = []
    for i in range(numberOfDays, len(testColumn) + numberOfDays):
        test_features.append(test_inputs[i - numberOfDays: i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(
        test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def accuracyCal(predictions, actualValues):
    '''This function evaluates the Mean Absolute Error (MAE).

    predictions: predicted stock price
    actualValues: actual stock price'''

    error = np.absolute(predictions - actualValues)
    errorSum = np.sum(error, axis=0) / len(error)
    return errorSum[0]


def plotPredictions(predictions, actualValues, stock):
    '''This function plots the graph for both predicted and 
    actual stock prices of a given company.

    predictions: predicted stock price of a company
    actualValues: actual stock price of a company
    stock: the company's stock symbol based on Yahoo Finance'''

    plt.figure(figsize=(10, 6))
    plt.plot(actualValues, color='blue',
             label='Actual {} Stock Price'.format(stock))
    plt.plot(predictions, color='red',
             label='Predicted {} Stock Price'.format(stock))
    plt.title('{} Stock Price Prediction'.format(stock))
    plt.xlabel('Date')
    plt.ylabel('{} Stock Price'.format(stock))
    plt.legend()
    plt.show()


def main():
    '''Main function implements the "best" model to train the final training
    set in order to predict the stock prices of various companies for the 
    month of November, 2019.'''

    numLayers = 1
    numNodes = 100
    numDays = 60
    # Put Yahoo Finance based stock symbol here
    stock = 'AAPL'  # Yahoo Finance based stock symbol for Apple

    trainData = pd.read_csv('{}_train.csv'.format(stock))
    trainDataProcessed = trainData.iloc[:, 1:2].values
    trainDataScaled = scaler.fit_transform(trainDataProcessed)

    inputsData = []
    labels = []

    for i in range(numDays, len(trainDataScaled)):
        inputsData.append(trainDataScaled[i - numDays:i, 0])
        labels.append(trainDataScaled[i, 0])

    inputsData, labels = np.array(inputsData), np.array(labels)
    inputsData = np.reshape(
        inputsData, (inputsData.shape[0], inputsData.shape[1], 1))

    model = Sequential()
    createNN(model, numLayers, numNodes, 0.2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(inputsData, labels, epochs=100, batch_size=32)

    testData = pd.read_csv('{}_test.csv'.format(stock))
    testColumn = testData.iloc[:, 1:2].values
    predictions = predictData(trainData, testData, numDays, model)
    errorSum = accuracyCal(predictions, testColumn)
    plotPredictions(predictions, testColumn, stock)


if __name__ == "__main__":
    main()
