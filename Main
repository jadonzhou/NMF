import numpy as np  
from sklearn.decomposition import NMF 
from sklearn.decomposition import TruncatedSVD            
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV


dataraw = pd.read_csv('/Users/jadonzhou/Research Projects/OneDrive/K Man HKU/HCC rec project/Lasso analysis/Database.csv')
# variables=['EGF', 'FGF2',
#        'EOTAXIN', 'TGFa', 'GCSF', 'FLT3L', 'GMCSF', 'FRACTALKINE', 'IFNa2',
#        'IFNg', 'GRO', 'IL10', 'MCP3', 'IL12P40', 'MDC',
#        'IL12P70', 'PDGFAA', 'IL13', 'PDGFAB_BB', 'IL15', 'SCD40L', 'IL17A',
#        'IL1RA', 'IL1a', 'IL9', 'IL1b', 'IL2',
#        'IL4', 'IL5', 'IL6', 'IL7', 'IL8', 'IP10', 'MCP1', 'MIP1a', 'MIP1b',
#        'RANTES', 'TNFa', 'TNFb', 'VEGF', 'EVENTP']
variables=['EOTAXIN', 'IFNa2','IL17A']
featuredata=np.array(dataraw[variables])
#min_max_scaler = preprocessing.MinMaxScaler()
#featuredatascale = min_max_scaler.fit_transform(featuredata)
data=np.transpose(featuredata)

nmf_model = NMF(n_components=3) # 设有2个主题
core = nmf_model.fit_transform(data)
latentvectors = nmf_model.components_
latentvectors=latentvectors.T
latentvectors=pd.DataFrame(latentvectors)
Data=dataraw[variables]
Data[latentvectors.columns] = latentvectors
Data.to_csv('/Users/jadonzhou/Research Projects/OneDrive/K Man HKU/HCC rec project/Lasso analysis/Database_latent_new.csv')
print('User distribution：')
print(latentvectors)
print('Core distribution：')
print(core)


# Optimal NMF rank selection
from sklearn import decomposition, datasets, model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
dataraw = pd.read_csv('/Users/jadonzhou/Research Projects/OneDrive/K Man HKU/HCC rec project/Lasso analysis/Database.csv')
#variables=['EOTAXIN', 'IFNa2','IL5','IL17A']
featuredata=np.array(dataraw[variables])
# performance evaluation
def get_score(model, data, scorer=metrics.explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)
# select optimal rank number
ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
perfs_train = []
scaler = preprocessing.StandardScaler(with_mean=False).fit(featuredata)
for k in ks:
    nmf = decomposition.NMF(n_components=k).fit(scaler.transform(featuredata))
    perfs_train.append(get_score(nmf, scaler.transform(featuredata)))
print(perfs_train)
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.plot(ks, perfs_train, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
ax1.set_xticks(np.arange(len(ks)+1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('Factorization rank number (Number of latent variables)') 
plt.ylabel('Total explained variance')
plt.annotate('{:,.2%}'.format(perfs_train[0]), (1, perfs_train[0]), fontsize=12)
plt.annotate('{:,.2%}'.format(perfs_train[1]), (2, perfs_train[1]), fontsize=12)
plt.annotate('{:,.2%}'.format(perfs_train[2]), (3, perfs_train[2]), fontsize=12)
plt.show()

