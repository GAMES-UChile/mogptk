import numpy as np
import pandas as pd

"""
Merge currency exchange and inflation into single dataset
"""

# load currency exchange
df = pd.read_csv('currency_exchange.csv')

# obtain absolute day starting with monday = 1
df['Day'] = df['Jul.Day'] - df['Jul.Day'].min() + 2
df['YYYY/MM/DD'] = pd.to_datetime(df['YYYY/MM/DD'])

# load second dataset
inflation_df = pd.read_csv('inflation.csv')
# transform to datetime
inflation_df['TIME'] = pd.to_datetime(inflation_df['TIME'])

# get date range
idx = (inflation_df['TIME'] >= np.datetime64('2017-01-01')) & (inflation_df['TIME'] <= np.datetime64('2018-12-31'))
# only inflation in usa
inflation_usa = inflation_df.loc[idx, :].loc[inflation_df['LOCATION']=='USA'][['TIME', 'Value']]

# set the first date to match other dataset
inflation_usa['TIME'].iloc[0] = np.datetime64('2017-01-03')
# set new index
inflation_usa.set_index('TIME', inplace=True)

# expand index
inflation_usa = inflation_usa.reindex(pd.date_range(start='2017-01-03', end='2018-12-31'), method='ffill')
# rename col
inflation_usa.rename(columns={'Value':'CPI USA'}, inplace=True)


# final dataframe
final_df = inflation_usa.loc[df['YYYY/MM/DD']]
var_list = ['Day',
          'CAD/USD',
          'EUR/USD',
          'JPY/USD',
          'GBP/USD',
          'CHF/USD',
          'AUD/USD',
          'HKD/USD',
          'NZD/USD',
          'KRW/USD',
          'MXN/USD']

for var in var_list:
    final_df[var] = df[var].values
    
final_df = final_df[['Day',
                     'CPI USA',
                     'CAD/USD',
                     'EUR/USD',
                     'JPY/USD',
                     'GBP/USD',
                     'CHF/USD',
                     'AUD/USD',
                     'HKD/USD',
                     'NZD/USD',
                     'KRW/USD',
                     'MXN/USD']]
final_df.to_csv('final_dataset.csv')