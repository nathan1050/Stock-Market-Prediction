import tkinter
from tkinter import *
from datetime import date
import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import tkinter
from tkinter import *
from datetime import date
import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
import datetime

window = tkinter.Tk()

window.title("Stock Market Prediction using Neural Networks")
l = Label(window, text="Enter ticker of the stock for prediction")
l.grid(row = 0, column = 0)

e = Entry(window, width = 50)
e.grid(row = 0, column = 3)
message = e.get()

#start date

#end date 

#print(message)
def clicked():
    message = e.get()
    print("The entered tiker is:")
    print(message)
    #not in required format 
    today = date.today()
    day2 = today + datetime.timedelta(days=1)
    print("Today's date:", today)
    #in the required format
    d1 = today.strftime("%Y-%m-%d")
    print("d1 =", d1)
    #lable_text.set("")
    #l2 = Label (window, text = "The button is clicked ")
    #l2.grid(row = 1, column = 3)
    #df = quandl.get("WIKI/AMZN", start_date="2018-12-31", end_date="2019-12-31")
    # Take a look at the data

    df = si.get_data(message)
    print (df)
    df.fillna(df.mean(), inplace=True)
    # Get the Adjusted Close Price 
    df = df[['close']]
    print("ADJ CLOSE")
    print (df)
    # Take a look at the new data 
    #print(df.head())
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['close']]
    #df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
    print("Predictions Column")
    print(df)
    #print the new data set
    #print(df.tail())
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))
    print("################################# X ##################")
    print(X)
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    print("################################# Y ##################")
    print(y)
    # Get all of the y values except the last '30' rows
    y1 = y[:-forecast_out]
    print("################################# Y ##################")
    print(y1)

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    #Remove the last '30' rows
    X = X[:-forecast_out]
    print(X)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)
    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    print ("################################ FORECAST ###########################")
    print(x_forecast)

    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)
    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    print(svm_prediction)
    
    height = 5
    width = 3
    for i in range(5,12): #Rows
        for j in range(width): #Column
            #row 1 
            b = Label(window, text="Date")
            b.grid(row=5, column=0)
            
            b = Label(window, text="Price")
            b.grid(row=5, column=1)
            
            b = Label(window, text="Ticker")
            b.grid(row=5, column=2)

            #row 2 
            b = Label(window, text=today)
            b.grid(row=6, column=0)

            b = Label(window, text=lr_prediction[29])
            b.grid(row=6, column=1)

            b = Label(window, text=message)
            b.grid(row=6, column=2)

            #row 3
            b = Label(window, text=today + datetime.timedelta(days=1))
            b.grid(row=7, column=0)

            b = Label(window, text=lr_prediction[27])
            b.grid(row=7, column=1)

            b = Label(window, text=message)
            b.grid(row=7, column=2)

            #row 4
            b = Label(window, text=today + datetime.timedelta(days=2))
            b.grid(row=8, column=0)

            b = Label(window, text=lr_prediction[26])
            b.grid(row=8, column=1)

            b = Label(window, text=message)
            b.grid(row=8, column=2)

            #row 5
            b = Label(window, text=today + datetime.timedelta(days=3))
            b.grid(row=9, column=0)

            b = Label(window, text=lr_prediction[25])
            b.grid(row=9, column=1)

            b = Label(window, text=message)
            b.grid(row=9, column=2)

            #row 6
            b = Label(window, text=today + datetime.timedelta(days=4))
            b.grid(row=10, column=0)

            b = Label(window, text=lr_prediction[24])
            b.grid(row=10, column=1)

            b = Label(window, text=message)
            b.grid(row=10, column=2)

            #row 7
            b = Label(window, text=today + datetime.timedelta(days=5))
            b.grid(row=11, column=0)

            b = Label(window, text=lr_prediction[23])
            b.grid(row=11, column=1)

            b = Label(window, text=message)
            b.grid(row=11, column=2)

            #row 8
            b = Label(window, text=today + datetime.timedelta(days=5))
            b.grid(row=12, column=0)

            b = Label(window, text=lr_prediction[22])
            b.grid(row=12, column=1)

            b = Label(window, text=message)
            b.grid(row=12, column=2)

            



    
b = Button(window, text = "Submit", command=clicked)
b.grid(row = 2, column = 2)

window.mainloop()
