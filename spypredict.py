__author__ = 'bryantb'
import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
from sklearn.qda import QDA
import re
from dateutil import parser
from backtest import Strategy, Portfolio

def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  pd.io.data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

    return df

def getStockFromQuandl(symbol, name, start, end):
    """
    Downloads Stock from Quandl.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    import Quandl
    df =  Quandl.get(symbol, trim_start = start, trim_end = end, authtoken="your token")

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + name
    df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()

    return df

def getStockDataFromWeb(fout, start_string, end_string):
    """
    Collects predictors data from Yahoo Finance and Quandl.
    Returns a list of dataframes.
    """
    start = parser.parse(start_string)
    end = parser.parse(end_string)

    nasdaq = getStock('^IXIC', start, end)
    frankfurt = getStock('^GDAXI', start, end)
    london = getStock('^FTSE', start, end)
    paris = getStock('^FCHI', start, end)
    hkong = getStock('^HSI', start, end)
    nikkei = getStock('^N225', start, end)
    australia = getStock('^AXJO', start, end)

    djia = getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start_string, end_string)

    out =  pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()

    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]

def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n.
    - given AdjClose_* computes its moving average on n days

    """

    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)

    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

def applyRollMeanDelayedReturns(datasets, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    for dataset in datasets:
        columns = dataset.columns
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)

    return datasets

def mergeDataframes(datasets, index, cut):
    """
    merges datasets in the list
    """
    subset = []
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]

    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left')
    finance = finance[finance.index > cut]
    return finance

def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """

    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]

def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder()

    dataset['UpDown'] = dataset['Return_Out']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)

    features = dataset.columns[1:-1]
    X = dataset[features]
    y = dataset.UpDown

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]

    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    return X_train, y_train, X_test, y_test

def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """
    performs classification on daily returns using several algorithms (method).
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """

    if method == 'RF':
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'SVM':
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'GTB':
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'QDA':
        return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    SVM binary Classification
    """
    c = parameters[0]
    g =  parameters[1]
    clf = SVC()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Ada Boosting binary Classification
    """
    n = parameters[0]
    l =  parameters[1]
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Quadratic Discriminant Analysis binary Classification
    """
    def replaceTiny(x):
        if (abs(x) < 0.0001):
            x = 0.0001

    X_train = X_train.apply(replaceTiny)
    X_test = X_test.apply(replaceTiny)

    clf = QDA()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters):
    """
    Performs Feature selection for a specific algorithm
    """

    for maxlag in range(3, maxlags + 2):
        lags = range(2, maxlag)
        print ''
        print '============================================================='
        print 'Maximum time lag applied', max(lags)
        print ''
        for maxdelta in range(3, maxdeltas + 2):
            datasets = loadDatasets(path_datasets, fout)
            delta = range(2, maxdelta)
            print 'Delta days accounted: ', max(delta)
            datasets = applyRollMeanDelayedReturns(datasets, delta)
            finance = mergeDataframes(datasets, 6, cut)
            print 'Size of data frame: ', finance.shape
            print 'Number of NaN after merging: ', count_missing(finance)
            finance = finance.interpolate(method='linear')
            print 'Number of NaN after time interpolation: ', count_missing(finance)
            finance = finance.fillna(finance.mean())
            print 'Number of NaN after mean interpolation: ', count_missing(finance)
            finance = applyTimeLag(finance, lags, delta)
            print 'Number of NaN after temporal shifting: ', count_missing(finance)
            print 'Size of data frame after feature creation: ', finance.shape
            X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)

            print performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
            print ''
            