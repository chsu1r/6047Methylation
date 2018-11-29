import pandas as pd
import sklearn
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression

def parta():
    listy = []
    # chunks=pd.read_table('gdac.broadinstitute.org_LAML.Merge_methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.Level_3.2016012800.0.0/LAML.methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt',chunksize=1000000)
    # data = pandas.read_table()
    # data = data.set_index('Patient')
    # data = data.set_index('Patient')

    chunksize = 4500
    for chunk in pd.read_table('LAMLmethy.txt', chunksize=chunksize):
        cg = chunk["Hybridization REF"]
        chunk = chunk[chunk.columns[1::4]]  # isolate beta values for each participant
        chunk["Hybridization REF"] = cg
        chunk = chunk.set_index("Hybridization REF")
        if "Composite Element REF" in chunk.index:
            chunk = chunk.drop("Composite Element REF", axis=0)
        listy.append(chunk.apply(pd.to_numeric).round())
    data = pd.concat(listy).dropna()

    # print(len(data))
    # print(data.columns)
    # print(len(data.columns))
    dataT = data.T
    # print(len(dataT))
    # print(dataT.columns)

    X, y = make_regression(n_features=2, random_state=0)
    regr = ElasticNetCV(cv=5, random_state=0)
    regr.fit(dataT, np.ones(len(dataT)))
    print(regr.alpha_) 
    print(regr.intercept_)
    print(regr.predict([[0, 0]]))  
   


    
    
parta()
