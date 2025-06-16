import pandas as pd
import pipe

df = pd.read_csv('Rail_Ridership_Dataset.csv')

df.columns = ['Month', 'Ridership']
df['MonthIndex'] = df['Month'].str.extract('(\d+)').astype(int)

x = df['MonthIndex']
y = df['Ridership']

pipe.process(x, y)