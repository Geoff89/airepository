# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:39:22 2017

@author: jeffnerd
"""

index = pd.date_range('1/1/2000',periods = 8)
s = pd.Series(np.random.randn(5), index = ['a','b','c','d','e'])
df = pd.DataFrame(np.random.randn(8,3), index=index, columns = ['A','B','C'])
wp = pd.Panel(np.random.randn(2,5,4), items=['Item1','Item2'],
              major_axis = pd.date_range('1/1/2000',periods=5),
              minor_axis = ['A','B','C','D'])
p4d = pd.Panel4D(np.random.randn(2, 2, 5, 4),
                 labels=['Label1','Label2'],
                 items=['Item1', 'Item2'],
                 major_axis=pd.date_range('1/1/2000', periods=5),
                 minor_axis=['A', 'B', 'C', 'D'])


#output of series

a    0.387195
b   -0.759060
c    0.370356
d    0.117681
e   -0.452349
dtype: float64
#output of dataframe
A         B         C
2000-01-01 -1.137598  1.348027  1.413485
2000-01-02 -1.116438 -1.036772  0.872411
2000-01-03 -0.581985  0.210275  0.382482
2000-01-04  1.467543  0.068101  0.027009
2000-01-05  0.408531  0.548986 -0.018922
2000-01-06  1.632067 -0.398663 -1.959425
2000-01-07  0.559963  1.003821  0.189993
2000-01-08  0.417329 -0.063453  0.946255

#output of panel
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 5 (major_axis) x 4 (minor_axis)
Items axis: Item1 to Item2
Major_axis axis: 2000-01-01 00:00:00 to 2000-01-05 00:00:00
Minor_axis axis: A to D

#output of panel4d
<class 'pandas.core.panelnd.Panel4D'>
Dimensions: 2 (labels) x 2 (items) x 5 (major_axis) x 4 (minor_axis)
Labels axis: Label1 to Label2
Items axis: Item1 to Item2
Major_axis axis: 2000-01-01 00:00:00 to 2000-01-05 00:00:00
Minor_axis axis: A to D

#creating the pandas as pd
dates = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C',
'D'])
#panel
panel = pd.Panel({'one' : df, 'two' : df - df.mean()})
#output of it
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 8 (major_axis) x 4 (minor_axis)
Items axis: one to two
Major_axis axis: 2000-01-01 00:00:00 to 2000-01-08 00:00:00
Minor_axis axis: A to D

df.loc[:,['B', 'A']] = df[['A', 'B']]

#the corect way is to use raw values
df.loc[:,['B', 'A']] = df[['A', 'B']].values

# attribute access
# you can accessYou may access an index on a Series, column 
# on a DataFrame, and an item on a Panel directly as an attribute
sa = pd.Series([1,2,3],index=list('abc'))
dfa = df.copy()
#accessing them as a data attribute
sa.b
dfa.A
panel.one

#you can use attribute access to modify an existing element of a series or
#column on a DataFrame
sa.a = 5
dfa['A'] = list(range(len(dfa.index))) # use this form to create a new
˓→column preffered to dfa.A = list(range(len(dfa.index)))

#DAtaframe 2
dfl = pd.DataFrame(np.random.randn(5,4), columns=list('ABCD'), 
                   index=pd.date_range('20130101',periods=5))
s1 = pd.Series(np.random.randn(6),index=list('abcdef'))
#slicing using loc
dfl.loc['20130102':'20130104']
s1.loc['c']
s1.loc['c':]
s1.loc['c':] = 0
 #dataframe2 for loc accessing
df1 = pd.DataFrame(np.random.randn(6,4), index=list('abcdef'), 
                    columns=list('ABCD'))
df1.loc['a'] > 0 #for boolean values
df1.loc[:, df1.loc['a'] > 0]
df1.loc['a', 'A']
df1.loc[['a', 'b', 'd'], :]
df1.loc[:,['A','B','D']]
df1.loc['d':'A':'C']

#for interger access using iloc
s1 = pd.Series(np.random.randn(5), index=list(range(0,10,2)))
s1.iloc[:3]
s1.iloc[3]
s1.iloc[:3] = 0

#for dataframe
df1 = pd.DataFrame(np.random.randn(6,4), index=list(range(0,12,2)),
                   columns=list(range(0,8,2)))
df1.iloc[:3]
df1.iloc[1:5, 2:4]
df1.iloc[[1, 3, 5], [1, 3]]
df1.iloc[1:3, :]
df1.iloc[:, 1:3]
df1.iloc[1, 1]
df1.iloc[1]

#using callables to index
df1 = pd.DataFrame(np.random.randn(6, 4),index=list('abcdef'),
                   columns=list('ABCD'))
df1.loc[lambda df: df.A > 0, :]
df1.loc[:, lambda df: ['A', 'B']]
df1.iloc[:, lambda df: [0, 1]]
df1[lambda df: df.columns[0]]
#You can use callable indexing in Series.
df1.A.loc[lambda s: s > 0]