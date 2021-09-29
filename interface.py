import tkinter
from tkinter import *

window = tkinter.Tk()

window.title("Stock Market Prediction ")
l = Label(window, text="Enter Ticker")
l.grid(row = 0, column = 0)

l1 = Label(window, text = "Enter Date")
l1.grid(row = 3, column = 0)

e = Entry(window, width = 50)
e.grid(row = 0, column = 3)
ticker = e.get()

e1 = Entry(window, width = 50)
e1.grid(row = 3, column = 3)
ticker = e1.get()

def clicked():
    ticker = e.get()
    year = e1.get()
    print(ticker)
    print(year)
    #lable_text.set("")
    
b = Button(window, text = "Get Updates", command=clicked)
b.grid(row = 6, column = 3)
window.mainloop()
