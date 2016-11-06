__author__ = 'bryantb'

import datetime



from pandas_datareader import data, wb

def getstockdata(symbol, strtyear, strtmonth, strtday, endyear, endmonth, endday, datafile):
 start = datetime.datetime(strtyear, strtmonth, strtday)
 end = datetime.datetime(endyear, endmonth, endday)

 path='/Users/bryantb/PycharmProjets/spyprediction/data'

 sp = data.get_data_yahoo(symbol, start, end)

 sp.columns.values[-1] = 'AdjClose'
 sp.columns = sp.columns + '_SP500'
 sp['Return_SP500'] = sp['AdjClose_SP500'].pct_change()

 sp.to_csv('/Users/bryantb/PycharmProjects/spyprediction/data/'+datafile)

if __name__ == "__main__":
    #getstockdata('^GSPC', 2010, 1, 1, 2016, 6, 30, 'sp2.csv')