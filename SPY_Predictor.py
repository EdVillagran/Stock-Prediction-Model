import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests

#Function used to plot both dataframes
def plotBoth(original, predicted, forecasted):

    #Only plot days chose to be forecasted out
    original=original[-forecasted:]

    #Get both indexes to match so we can copy dates over
    original.index = predicted.index
    dates=original[["Date"]]
    predicted['Date']=dates.copy()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(original['Date'], original['Adj Close'], label="Original")
    plt.plot(predicted['Date'], predicted[0], label="Predicted")
    plt.legend()
    plt.show()


#Builds ML Model, uses Linear Regression Forecasting to make predictions
def LRFP(df, forecast, filename):
    
    # Only Going to use the Adj Close since that's the parameter we want to predict
    df = df[['Adj Close']]
    
    #Make a copy to hold as original dataframe
    original=df 
    
    # Create another column for target
    df['Prediction'] = df[['Adj Close']].shift(-forecast)

    # Create the independent data set x & Convert the dataframe to a numpy array
    x = np.array(df.drop(['Prediction'], 1))

    # Remove the last n rows We are Trying to predict
    x = x[:-forecast]

    # Create the dependent data set y & Convert the dataframe to a numpy array
    y = np.array(df['Prediction'])

    # Remove Last 30 we want to predict
    y = y[:-forecast]

    # Train and Split the data into 90% training and 10% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

    #Create a pickle file for that stock to save the model with best score
    #replaces name.csv with name.pickle
    picklefile=filename[:-3]+"pickle"

    #Create a variable to compare best accuracy score
    best = 0
    
    #Do n-number of runs to and pick the one with the best score
    for _ in range(100):
        
        # Train and Split the data into 90% training and 10% testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

        # Create and train the Linear Regression  Model
        linear = LinearRegression()
        linear.fit(x_train, y_train)
        
        #Get Accuracy score for comparison
        accuracy = linear.score(x_test, y_test)

        # Only save model if its better
        if accuracy > best:
            best = accuracy
            # Create and save a model
            with open(picklefile, "wb") as f:
                pickle.dump(linear, f)
                
    #open pickle file
    pickle_in = open(picklefile, "rb")
    linear = pickle.load(pickle_in)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast:]

    # Print linear regression model predictions for the next N days
    lr_prediction = linear.predict(x_forecast)


    #Write data to stockpredictions.txt
    with open("stockprediction.txt", "w+") as f:
        for i in lr_prediction:
            f.write(str(i)+"\n")


    for x, y in enumerate(lr_prediction):
        print("Day", x + 1, "Prediction", "%.2f" % y)

    df2 = pd.DataFrame(lr_prediction)
    return df2

'''
 Function downloadStockData recieves stock symbol input and uses the requests module to fetch 
 and download all data available from Jan. 01 2000 - Aug. 02 2020.
 Then creates a csv file to be used and returns the file name.
 If the stock symbol is invalid it will name the file NA.
 This was one way I thought of checking for a bad symbol. 
'''
def downloadStockData():

    stock=input("Enter Company Ticker Symbol (EX: Facebook- FB): ")
    
    #Ensure Uppercase
    stock=stock.upper()

    #url set to download available stock info from Jan 1 2000 - Aug 02, 2020
    url="https://query1.finance.yahoo.com/v7/finance/download/"+stock+"?period1=946684800&period2=1596412800&interval=1d&events=history"
    
    r = requests.get(url, allow_redirects=True)
    
    #If doesnt exist name file NA
    if r.status_code==404:
        filename="NA"

    else:
        filename=stock+'.csv'
        open(filename, 'wb').write(r.content)

    #returns name of file
    return filename

def main():

    filename=downloadStockData()
    
    #Check if stock ticker symbol is incorrect. By reading the filename, instead of the
    #file itself. 
    if filename=="NA":
        print("Invalid Symbol")

    else:

        df = pd.read_csv(filename)
        days=int(input("Enter number of days to forecast: "))
        predicted_df=LRFP(df, days,filename)
        plotBoth(df,predicted_df,days)

#Call main
if __name__=="__main__":
    main()

