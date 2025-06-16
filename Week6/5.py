import pandas as pd
import pipe

df = pd.read_csv('Library_Visitors_Dataset.csv')

df.columns = ['Month', 'Visitors']
df['MonthIndex'] = df['Month'].str.extract('(\d+)').astype(int)

x = df['MonthIndex']
y = df['Visitors']

pipe.process(x, y)