# StockPrediction
#Siang and Kunsang

To run the first (main) program, it will suffice to simply run the command:

python stockLSTM.py

Running this program will train the opening price data of Apple, Google, and Amazon from January 2014 to October 2019. Our “best” model will be used to predict the opening stock prices for November 2019 of each company. The program will also display three graph plots of predicted (red) and actual (blue) opening stock prices of each company for the month of November.

To run the second program, it will suffice to run the command:

python timeseriesCV.py

This program was used to find the best model for stock prediction. The lists of hidden nodes and layers in the program resulted in our best model -- which was 100 hidden nodes in 1 layer.

(You can change the number of hidden nodes and hidden layers to save running time)

