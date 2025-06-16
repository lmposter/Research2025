import pandas as pd
import pipe

df = pd.read_csv('Industrial_Output_Dataset.csv')

df.columns = ['Quarter', 'Output']
df['QuarterIndex'] = df['Quarter'].str.extract('(\d+)').astype(int)

x = df['QuarterIndex']
y = df['Output']

pipe.process(x, y)