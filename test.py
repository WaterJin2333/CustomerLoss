__author__ = 'Alex'

import pandas as pd
from Pareto_NBD import *

# df = pd.DataFrame.from_csv('cdnow_data.csv')
# freq = df.p1x
# rec = df.t_x
# age = df['T']

df=pd.read_excel('E:\\学习\\课程设计\\shengxian\\new_new_data_2.xlsx')
df = df.loc[df['each_cost']>1]
freq=(df['total_cost']/df['each_cost']).apply(lambda x:int(x))
print(freq)
rec=df['recent']
print(rec)
age=df['first']
print(age)
for i in rec.index:
    if rec.loc[i]-age.loc[i]>0:
        rec.loc[i]=age.loc[i]
        print(age.loc[i])
        print(rec.loc[i])
        print(i)
my_fit = ParetoNBD()
my_fit.fit(freq, rec, age)
df['p_alive'] = my_fit.p_alive(freq, rec, age)
print(df.to_string())



