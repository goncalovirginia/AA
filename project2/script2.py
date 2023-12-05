### Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

### Pandas
import pandas as pd

#Por causa do kmeans ig
import os
os.environ["OMP_NUM_THREADS"] = '1'

#########################################################
#                   Data Arrangement
#########################################################

test_data = pd.read_csv('project2/Data/test_data.csv')
train_data = pd.read_csv('project2/Data/train_data.csv')

#Drop ID
test_id = test_data['Id']
test_data.drop('Id', axis='columns', inplace=True)
train_data.drop('Id', axis='columns', inplace=True)

#[Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse',
#'SurvivalTime', 'Censored']
data_columns = list(test_data.columns)
labelled_columns = ['Age', 'Gender', 'Stage', 'TreatmentType']
unlabelled_columns = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

x_columns = ['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']
y_columns = ['Censored', 'SurvivalTime']

x_train = train_data[x_columns]
y_train = train_data[y_columns]


"""
#########################################################
#              MISSING VALUES IMPUTATION
#########################################################
"""
# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

"""
###############
# SimpleImputer
###############
"""

df = x_train.copy()
impute_mostfreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_MostFreq = impute_mostfreq.fit_transform(df)

"""
################
# KNN Imputer
################
"""

# K Nearest Neighbors
df = x_train.copy()
impute_knn = KNNImputer(missing_values=np.nan, n_neighbors=2, weights='distance')


#SurivalTime Imputation
df_aux = pd.DataFrame(df_MostFreq)
df_aux.columns = test_data.columns
df_aux['SurvivalTime'] = y_train['SurvivalTime']
df_aux['Censored'] = y_train['Censored']

df_MostFreq_x_KNN = pd.DataFrame(impute_knn.fit_transform(df_aux))
df_MostFreq_x_KNN.columns = train_data.columns


"""
#############################
#       NEW TRY
############################
"""

"""CMSE FUNCTION"""
def cMSE(y, y_hat, c):
  err = y-y_hat
  err = (1-c)*err**2 + c*np.maximum(0,err)**2
  return np.sum(err)/err.shape[0]

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler 
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA

#Dataframes
test_data = pd.read_csv('project2/Data/test_data.csv')
train_data = pd.read_csv('project2/Data/train_data.csv')
test_id = test_data['Id']
test_data.drop('Id', axis='columns', inplace=True)
train_data.drop('Id', axis='columns', inplace=True)

#column groups
test_columns = test_data.columns
labelled_columns = ['Age', 'Gender', 'Stage', 'TreatmentType']
unlabelled_columns = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

#create the transformers
columns_to_impute_knn = ['SurvivalTime']
knn_imputer = ('knn_imputer', KNNImputer(), columns_to_impute_knn)

columns_to_impute_most_frequent = unlabelled_columns
most_frequent_imputer = ('most_frequent_imputer', SimpleImputer(strategy='most_frequent'), columns_to_impute_most_frequent)

#create the column transformer
transformer = ColumnTransformer(
    transformers=[most_frequent_imputer, knn_imputer],
    remainder='passthrough'  # This will keep any columns not explicitly transformed
)

#fills the missing values
filled_data = transformer.fit_transform(train_data)

#Concatenate the remaining column names
columns = unlabelled_columns + ['SurvivalTime'] + labelled_columns + ['Censored']
train_data = pd.DataFrame(filled_data, columns=columns)

#Reorder
train_data = train_data[train_data.columns]

"""
#################################
#       Clustering Try
#################################
"""

from sklearn.cluster import KMeans

def plot_clusters(X,y_pred,title=''):
    """Plotting function; y_pred is an integer array with labels"""    
    plt.figure()
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.axis('equal')
    plt.show()

from itertools import combinations
feature_combinations = set(combinations(labelled_columns, 2))
"""
for (f1, f2) in feature_combinations:
    cluster_data = train_data[[f1,f2]].values
    kmeans = KMeans(n_clusters=4).fit(cluster_data)
    labels = kmeans.predict(cluster_data)
    plot_clusters(cluster_data, labels, f'KMeans4_{f1}_x_{f2}')
"""

def cluster_data(data, hasModel, cluster):
    #features = ['Age','Stage']
    #features = labelled_columns
    features = test_columns
    df = data[features].values
    if hasModel:
        kmeans_cluster = cluster
    kmeans_cluster = KMeans(n_clusters=4).fit(df)
    labels = kmeans_cluster.predict(df)
    #Plot only works for 2D array
    #plot_clusters(cluster_data, labels, f'Cluster')
    #print(f'Cluster Labels: {labels}')
    data['Cluster'] = labels
    if hasModel:
        return data
    return data, kmeans_cluster

##################################################################
c = train_data['Censored']
#Preparing the cross validation
stratkfold = StratifiedKFold(n_splits=5, shuffle=True)
cmse_scorer = make_scorer(cMSE, greater_is_better=False, c=c)

def model_testing(X, y, c, model, model_name, clustering):
    print(f'X shape: {X.shape}\n y shape: {y.shape}')
    scores = []
    for train_index, test_index in stratkfold.split(X,c):
        x_train_fold, y_train_fold = X.iloc[train_index], y.iloc[train_index]
        x_test_fold, y_test_fold = X.iloc[test_index], y.iloc[test_index]
        c_test_fold = c.iloc[test_index]

        model.fit(x_train_fold, y_train_fold)
        y_hat_fold = model.predict(x_test_fold)
        scores.append(cMSE(y_test_fold, y_hat_fold, c_test_fold))
    
    if clustering:
        print("With Clustering:\n")
    #Cross Validation Scores
    print(f'Cross Validation - {model_name}:\n')
    print(f'\t scores: {scores}\n')
    cmse = sum(scores) / len(scores)
    print(f'\t mean: {cmse}\n')
    
    return (cmse, model)

def gen_prediction(model, model_name, x_test, cmse, clustering):
    mostFreq = transformer.named_transformers_['most_frequent_imputer']
    x_test_filled = mostFreq.fit_transform(x_test)
    x_test_filled = pd.DataFrame(x_test_filled, columns=x_test.columns)
    if clustering:
        model_name = model_name
        x_test_filled_and_clustered = cluster_data(x_test_filled, True, kmeans_cluster)
        y_prediction = model.predict(x_test_filled_and_clustered)
    else:
        y_prediction = model.predict(x_test_filled)
    
    prediction = pd.DataFrame()
    prediction['id'] = test_id
    prediction['TARGET'] = y_prediction
    prediction.to_csv(f'project2/predicts/{model_name}_CMSE-{round(cmse, 5)}.csv', index=False)

"""Names"""
name_x = '_x_'
imputation_name = 'MostFreq_x_KNN'

"""Scalers"""
std, std_name = StandardScaler(), 'StdScaler'
min_max, mm_name = MinMaxScaler(), 'MinMaxScaler'
max_abs, ma_name = MaxAbsScaler(), 'MaxAbsScaler'

"""PCA"""
pca_name = 'PCA'

"""Regressions"""
lr_name = 'LinearReg'
ridge_name = 'RidgeReg'

#DOES NOT MAKE SENSE
#polys = [2,3,4,5]
#poly_name = 'PolyFeat'


def linear_model(steps, model_name, X_test, X, X_c, y, c):
    
    #Without Poly features
    name_1 = model_name + name_x + lr_name
    
    """Linear Regression"""
    steps_1 = steps + [(lr_name, LinearRegression())]
    model_1 = Pipeline(steps_1)
    (cmse_1, model_1) = model_testing(X, y, c, model_1, name_1, False)
    gen_prediction(model_1, name_1, X_test, cmse_1, False)

    
    """Linear Reg With clustering"""
    name_1_c = name_1 + '_x_Clustering'
    model_1_c = Pipeline(steps_1)
    (cmse_1_c, model_1) = model_testing(X_c, y, c, model_1_c, name_1, True)
    gen_prediction(model_1, name_1, X_test, cmse_1_c, True)


    return (cmse_1, name_1), (cmse_1_c, name_1_c)
    #Does not make sense since we are trying to reduce the dimensionality with PCA
    """Linear with Poly Feats"""
    """
    for n in polys:
        #Name
        poly_n_name = poly_name + f'-{n}'
        name_n = model_name + name_x + poly_n_name + name_x + lr_name
        #Polynomial Features
        steps_n = steps + [(poly_n_name, PolynomialFeatures(degree=n))]
        
        #Linear Regression
        steps_n = steps_n + [(lr_name, LinearRegression())]
        model_n = Pipeline(steps_n)
        (cmse_n, model_n) = model_testing(X, y, c, model_n, poly_n_name)
        gen_prediction(model_n, name_n, X_test, cmse_n, False)

        #Linear Reg With clustering
        model_n_c = Pipeline(steps_n)
        (cmse_n, model_n_c) = model_testing(X_c, y, c, model_n_c, poly_n_name)
        gen_prediction(model_n_c, name_n, X_test, cmse_n, True)
        """

#optimal -> 5 from previous observations
alphas = [10e-4, 10e-3, 10e-2, 1, 5, 10]

def ridge_model(steps, model_name, X_test, X, X_c, y, c):
    
    cmse_normal = []
    cmse_clustered = []
    
    for alpha in alphas:
        
        #Without Poly features
        name_n = model_name + name_x + ridge_name + f'-alpha-{alpha}'
        
        """Ridge Regression"""
        steps_n = steps + [(ridge_name, Ridge(alpha=alpha))]
        model_n = Pipeline(steps_n)
        (cmse_n, model_n) = model_testing(X, y, c, model_n, name_n, False)
        gen_prediction(model_n, name_n, X_test, cmse_n, False)
        
        cmse_normal.append((cmse_n, name_n))
        
        """Ridge Reg With clustering"""
        name_n_c = name_n + '_x_Clustering'
        model_n_c = Pipeline(steps_n)
        (cmse_n, model_n_c) = model_testing(X_c, y, c, model_n_c, name_n, True)
        gen_prediction(model_n_c, name_n_c, X_test, cmse_n, True)
        
        cmse_clustered.append((cmse_n, name_n_c))
        
    return cmse_normal, cmse_clustered

"""MODELS Init Names"""

model_std_name = imputation_name + name_x + std_name

model_mm_name = imputation_name + name_x + mm_name

model_ma_name = imputation_name + name_x + ma_name

"""Data"""
#Spliting the X, y and censored
x_columns = test_data.columns
X, y, c = train_data[x_columns], train_data['SurvivalTime'], train_data['Censored']


train_data, kmeans_cluster = cluster_data(train_data, False, 1)
#X with clustering
x_clustered = x_columns.append(pd.Index(['Cluster']))
X_c= train_data[x_clustered]


#PCA optimal n_components is 5
#pcas = [2,3,4,5]
#pcas = [5]
#for n in pcas:
n = 5
print(f"\n\n\n PCA {n}")
pca, pca_name = PCA(n_components=n), f'PCA-{n}'

steps_std = [(std_name, std)] + [(pca_name, pca)]
steps_mm = [(mm_name, min_max)] + [(pca_name, pca)]
steps_ma = [(ma_name, max_abs)] + [(pca_name, pca)]

"""Standard Scaler"""
model_name = model_std_name + name_x + pca_name
score_l_std, score_l_c_std = linear_model(steps_std, model_name, test_data, X, X_c, y, c)
scores_r_std, scores_r_c_std = ridge_model(steps_std, model_name, test_data, X, X_c, y, c)

"""MinMax Scaler"""
model_name = model_mm_name + name_x + pca_name
score_l_mm, score_l_c_mm = linear_model(steps_mm, model_name, test_data, X, X_c, y, c)
scores_r_mm, scores_r_c_mm = ridge_model(steps_mm, model_name, test_data, X, X_c, y, c)

"""MaxAbs Scaler"""
model_name = model_ma_name + name_x + pca_name
score_l_ma, score_l_c_ma = linear_model(steps_ma, model_name, test_data, X, X_c, y, c)
scores_r_ma, scores_r_c_ma =  ridge_model(steps_ma, model_name, test_data, X, X_c, y, c)
    

"""Plots"""

index = ['Linear_Default', 'Linear_With_Clustering']

df = pd.DataFrame({'StdScaler': [score_l_std[0], score_l_c_std[0]],
                   'MinMaxScaler': [score_l_mm[0], score_l_c_mm[0]],
                   'MaxAbsScaler': [score_l_ma[0], score_l_c_ma[0]]
                   }, index=index)
ax = df.plot.bar(rot=0)
ax.set(title ='Linear Reg CMSEs with PCA_5')
ax.legend(loc='lower left')
   
index = []
std_scores, mm_scores, ma_scores = [], [], []
std_scores_c, mm_scores_c, ma_scores_c = [], [], []
for i, alpha in enumerate(alphas):
    index = index + [f'\u03B1-{alpha}']
    std_scores.append(scores_r_std[i][0])
    std_scores_c.append(scores_r_c_std[i][0])
    mm_scores.append(scores_r_mm[i][0])
    mm_scores_c.append(scores_r_c_mm[i][0])
    ma_scores.append(scores_r_ma[i][0])
    ma_scores_c.append(scores_r_c_ma[i][0])
    

df = pd.DataFrame({'StdScaler': std_scores,
                   'MinMaxScaler': mm_scores,
                   'MaxAbsScaler': ma_scores
                   }, index=index)
ax = df.plot.bar(rot=0)
ax.set(title ='Ridge: CMSEs with PCA_5')
ax.legend(loc='lower left')

df = pd.DataFrame({'StdScaler': std_scores_c,
                   'MinMaxScaler': mm_scores_c,
                   'MaxAbsScaler': ma_scores_c
                   }, index=index)
ax = df.plot.bar(rot=0)
ax.set(title ='Ridge w/ Clustering: CMSEs with PCA_5')
ax.legend(loc='lower left')

"""
#model1 = make_pipeline(StandardScaler(), PCA(n_components=pca), LinearRegression())
model1_name = f'MostFreqImput_x_KNNImput_x_StdScaler_x_PCA-{pca}_x_LinearReg'
alpha = 0.1
#model2 = make_pipeline(StandardScaler(), PCA(n_components=pca), Ridge(alpha=alpha))
model2_name = f'MostFreqImput_x_KNNImput_x_StdScaler_x_PCA-{pca}_x_Ridge-alpha-{alpha}'
    

#without clustering
(cmse1, cmse2, model1, model2) = model_testing(X, y, c)
gen_prediction(model1, model1_name, test_data, cmse1, False)
gen_prediction(model2, model2_name, test_data, cmse2, False)
"""