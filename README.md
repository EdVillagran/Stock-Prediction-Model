# Stock Prediction Model </br>

This project began by me exploring Linear Regression use in Machine Learning and forcasting.
I began by running the closing cost of a csv file I had saved to my desktop, but wanted an easier way to read other stock data without having to 
download a new file everytime. I was able to to do so using the request library by figuering out how the yahoo finance history url worked.
By doing so you could enter any valid stock ticker symbol and it would download the data to the project folder in order to use. </br>

#### The structure of the program runs as followed:</br>
* Read user input for stock ticker symbol and days to forecast.
* Properly insert ticker symbol into url and download data as a csv file.
* Validate if symbol is valid or not by the status code given.
* If valid, create a pandas dataframe from csv file.
* Adjust the data frame to only hold adjusted closing cost. 
* Remove the last n-rows that will be forecasted.
* Create dependent dataset and train, split the data into 90% training and 10% testing.
* Create a pickle file file to store model with the best accuracy in n-number of runs.
* Each run we will train test split, create the linear regretion model and fit it.
* Only the model with the best accuracy score will be stored into the pickle file.
* Using the best model, forecast out the adj. closing cost for given days.
* Plot both original and forecasted prices.
