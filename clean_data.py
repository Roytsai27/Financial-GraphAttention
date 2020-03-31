import pandas as pd
import numpy as np
import math
import pickle
from sklearn.preprocessing import StandardScaler ,LabelEncoder


# the information of stocks(i.e. name & category of sector).
SP500_name = pd.read_csv("SP500_category.csv")
#SP500_name.head()

# original information of stock price. 
SP500_stock = {}
for target in SP500_name["company"]:
    da = {}
    da['category'] = SP500_name[SP500_name.company==target]['category'].iloc[0]
    da['stock_price'] = pd.read_csv("./SP500_dataset/%s.csv"%(target))
    SP500_stock[target] = da

# Let all stocks has the same date.
need_day = np.array(SP500_stock["BLL"]["stock_price"]['Date'])
for target in SP500_stock.keys():
    SP500_stock[target]["stock_price"] = SP500_stock[target]["stock_price"][SP500_stock[target]["stock_price"]['Date'].isin(need_day)].reset_index(drop=True)
    SP500_stock[target]["stock_price"].index = SP500_stock[target]["stock_price"]['Date']


### feature ###

# normalize stock price
normalize_scalar = {}
for target in SP500_stock.keys():
    scaler = StandardScaler()
    nor_data = scaler.fit_transform(np.array(SP500_stock[target]["stock_price"]["Close"]).reshape(-1,1)).ravel()
    SP500_stock[target]["stock_price"]["nor_close"] = nor_data
    normalize_scalar[target] = scaler

# calculate return ratio
for target in SP500_stock.keys():
    return_tratio = []
    data = np.array(SP500_stock[target]["stock_price"]["Close"])
    for i in range(len(data)):
        if i == 0:
            return_tratio.append(0)
        else:
            return_tratio.append((data[i]-data[i-1])/data[i-1])
    SP500_stock[target]["stock_price"]["return ratio"] = return_tratio


# feature of c_open / c_close / c_low
for target in SP500_stock.keys():
    function = lambda x,y: (x/y)-1
    data = SP500_stock[target]["stock_price"]
    data["c_open"] = list(map(function,data["Open"],data["Close"]))
    data["c_high"] = list(map(function,data["High"],data["Close"]))
    data["c_low"] = list(map(function,data["Low"],data["Close"]))


# 5 / 10 / 15 / 20 / 25 / 30 days moving average
for target in SP500_stock.keys():
    data = SP500_stock[target]["stock_price"]["Close"]
    for i in [5,10,15,20,25,30]:
        q = []
        for day in range(len(data)):
            if day >= i-1:
                q.append((np.mean(data.iloc[day-i+1:day+1])/data.iloc[day])-1)
            if day < i-1:
                q.append(0)
        SP500_stock[target]["stock_price"]["%s-days"%(i)] = q


# category of sector (one hot encoding)
label = LabelEncoder()
label.fit(SP500_name["category"].unique())

for target in SP500_stock.keys():
    for label in SP500_name["category"].unique():
        cate = SP500_stock[target]['category']
        if label != cate:
            SP500_stock[target]["stock_price"]["label_%s"%(label)] = 0
        if label == cate:
            SP500_stock[target]["stock_price"]["label_%s"%(label)] = 1


# total feature
features = {}
for target in SP500_stock.keys():
    features[target] = SP500_stock[target]["stock_price"].iloc[30:,7:].reset_index(drop=True)

# movement of stock 
Y_buy_or_not = {}
for target in SP500_stock.keys():
    Y_buy_or_not[target] = (features[target]['return ratio']>=0)*1            


## Trianing & Testing ##
train_size = 0.2
test_size = 0.8
days = len(features["BLL"])

train_day = int(days*train_size)

# data of training set and testing set
train_data = {}
test_data = {}
train_Y_buy_or_not = {}
test_Y_buy_or_not = {}

for i in SP500_stock.keys():
    train_data[i] = features[i].iloc[:train_day,:]
    train_Y_buy_or_not[i] = Y_buy_or_not[i][:train_day]
    test_data[i] = features[i].iloc[train_day:,:]
    test_Y_buy_or_not[i] = Y_buy_or_not[i][train_day:]


# week represents the number of our inputs
def before_day(week):
    # train
    train = {}
    for w in range(week):
        train_x = []
        for tr_ind in range(len(train_data["BLL"])-7-(week-2)-1):
            tr = []
            for target in SP500_stock.keys():
                data = train_data[target]
                tr.append(data.iloc[tr_ind+w:tr_ind+w+7,:].values)
            train_x.append(tr)
        train["x%s"%(w+1)] = np.array(train_x)
        
    train_y1,train_y2 = [] ,[]
    for tr_ind in range(len(train_data["BLL"])-7-(week-2)-1):
        all_stock_name = list(SP500_stock.keys())
        tr_y1 , tr_y2 = [] , []
        for target in SP500_stock.keys():
            data = train_data[target]
            tr_y1.append(data["return ratio"].iloc[tr_ind+(week-1)+7])
            tr_y2.append(train_Y_buy_or_not[target].iloc[tr_ind+(week-1)+7])
        train_y1.append(tr_y1)
        train_y2.append(tr_y2)
    train['y_return ratio'] = np.array(train_y1)
    train["y_up_or_down"]= np.array(train_y2)

    #test
    test = {}
    for w in range(week):
        test_x = []
        for te_ind in range(len(test_data["BLL"])-7-(week-2)-1):
            te = []
            for target in SP500_stock.keys():
                data = test_data[target]
                te.append(data.iloc[te_ind+w:te_ind+w+7,:].values)
            test_x.append(te)
        test['x%s'%(w+1)] = np.array(test_x)
    
    test_y1,test_y2 = [] ,[]
    for te_ind in range(len(test_data["BLL"])-7-(week-2)-1):
        te_y1 , te_y2 = [] , [] 
        for target in SP500_stock.keys():
            data = test_data[target]
            te_y1.append(data["return ratio"].iloc[te_ind+(week-1)+7])
            te_y2.append(test_Y_buy_or_not[target].iloc[te_ind+(week-1)+7])
        test_y1.append(te_y1)
        test_y2.append(te_y2)
    test['y_return ratio'] = np.array(test_y1)
    test["y_up_or_down"]= np.array(test_y2)
    
    data = {"train":train,"test":test}
    
    return(data)




