import pandas as pd
import pipe

df = pd.read_csv('Screen_Time_Dataset.csv')

df.columns = ['Week', 'ScreenTime']
df['WeekIndex'] = df['Week'].str.extract('(\d+)').astype(int)

x = df['WeekIndex']
y = df['ScreenTime']

pipe.process(x, y)

