import pandas as pd
import pipe

df = pd.read_csv('Electricity_Consumption_Dataset.csv')

df.columns = ['Month', 'Consumption']
df['MonthIndex'] = df['Month'].str.extract('(\d+)').astype(int)

x = df['MonthIndex']
y = df['Consumption']

pipe.process(x, y)