### Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
### Pandas
import pandas as pd

test_data = pd.read_csv('project2/Data/test_data.csv')
train_data = pd.read_csv('project2/Data/train_data.csv')
test_data.drop('Id',
  axis='columns', inplace=True)
train_data.drop('Id',
  axis='columns', inplace=True)

#['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
data_columns = list(train_data.columns)

train_data_done = train_data.drop(['GeneticRisk', 'ComorbidityIndex', 'SurvivalTime', 'Censored'], axis='columns').copy()

labelled_columns = ['Age', 'Gender', 'Stage', 'TreatmentType']
unlabelled_columns = [ele for ele in data_columns if ele not in labelled_columns]

#Drops rows with Nan values
#To define what columns to look forthe Nan values, use the subset parameter
#Example: df.dropna(subset=['a', 'b'])
#To select a tresholdf of minimum non-Nan values, use the thresh parameter
#Example: df.dropna(thresh=n)
labelled_train_data_without_nan = train_data_done.dropna(how='any', axis='rows')
labelled_test_data_without_nan = test_data.dropna(how='any', axis='rows')

X = labelled_train_data_without_nan.loc[:, labelled_train_data_without_nan.columns != 'TreatmentResponse']
y= labelled_train_data_without_nan['TreatmentResponse']


fig, axs =  plt.subplots(4, 9, figsize=(27, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for i in range(0,9):
    for j in range(0,9):
        if j > i:
            x = train_data.iloc[:, i]
            y = train_data.iloc[:, j]
            xlabel = data_columns[i]
            ylabel = data_columns[j]
            if i < 4:
                axs[i,j].scatter(x,y)
                #axs[i,j].set_title(xlabel + " x " + ylabel)
                axs[i, j].set_xlabel(xlabel)
                axs[i, j].set_ylabel(ylabel)
                axs[i,j].set_title("graph_" + str(i) + "_" + str(j))
            else:
                axs[7-i, 8-j].scatter(x,y)
                axs[7-i, 8-j].set_xlabel(xlabel)
                axs[7-i, 8-j].set_ylabel(ylabel)
                axs[7-i, 8-j].set_title("graph_" + str(i) + "_" + str(j))
            
        
fig.tight_layout()
plt.show()


fig, axs =  plt.subplots(3, 3, figsize=(12, 12))
for i in range(0,9):
    feature = data_columns[i]
    data = train_data.iloc[:, i].to_numpy()
    a = i // 3
    b = i % 3
    axs[a,b].hist(data, bins=10, edgecolor='black')
    axs[a,b].set_title('Distribution of ' + feature)
    
fig.tight_layout()
plt.show()

#train_data['Age'].plot(kind='box', figsize=(3, 12))
#train_data.drop('Age', axis='columns', inplace=True)
#train_data.plot(kind='box', figsize=(24, 12))

from pandas.plotting import scatter_matrix
scatter_matrix(train_data, alpha=0.5, figsize=(15,15), diagonal='kde')
#scatter_matrix(train_data, alpha=0.5, figsize=(15,10), diagonal='hist')



df = pd.DataFrame(train_data.dropna(how='any', axis='rows'))

df.drop(['Censored', 'SurvivalTime'], axis='columns', inplace=True)
print(train_data.dtypes)
from pandas.plotting import radviz
plt.figure(figsize=(8, 8))
radviz(df, 'TreatmentType', color=['red', 'blue'])
plt.title('Radviz: TreatmentType')
plt.show()
plt.figure(figsize=(8, 8))
radviz(df, 'Gender', color=['red', 'blue'])
plt.title('Radviz: Gender')
plt.show()
plt.figure(figsize=(8, 8))
radviz(df, 'Stage', color=['red', 'green', 'blue', 'purple'])
plt.title('Radviz: Stage')
plt.show()


# Define the age intervals
bins = [0, 50, 60, 70, 80, float('inf')]
labels = ['<50', '51-60', '61-70', '71-80', '81<']
# Create a new column with age intervals
df_aux = df.copy()  # Use df.copy() to create a copy of the DataFrame
df_aux['Age_Category'] = pd.cut(df_aux['Age'], bins=bins, labels=labels, right=False)
df_aux.drop('Age', axis='columns', inplace=True)

# Use Radviz to plot for Age
fig, ax = plt.subplots(figsize=(8, 6))
radviz(df_aux, class_column='Age_Category', ax=ax, color=['red', 'green', 'blue', 'cyan', 'orange'])
plt.title('Radviz: Age_Category')
plt.show()

# Define the survivaltime intervals
#bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, float('inf')]
#labels = ['<1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8<']
#df_aux = df.copy()  # Use df.copy() to create a copy of the DataFrame
# Create a new column with survival time intervals
#df_aux['SurvivalTime_Category'] = pd.cut(df_aux['SurvivalTime'], bins=bins, labels=labels, right=False)
# Use Radviz to plot for SurvivalTime
#fig, ax = plt.subplots(figsize=(8, 6))
#radviz(df_aux, class_column='SurvivalTime_Category', ax=ax)
#plt.show()

















