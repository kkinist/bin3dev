#!/usr/bin/python3
# get one row of a DataFrame into a Series and get its index
import pandas as pd
df = pd.DataFrame({'date': [20130101, 20130101, 20130102], 'location': ['a', 'a', 'c']})
print('whole df:\n', df)
df.iloc[0, 'date'] = 'changed'
print('df:\n', df)

