from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense, Dropout
import keras
import numpy as np
from keras.optimizers import Adam

import pandas as pd
from sklearn.model_selection import StratifiedKFold

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"

f = open("SignificantCpG_Variances",'r')
cpgSites = f.read()
cpgSites = cpgSites.split("\n")
cpgIndices = pd.Index(cpgSites[:-1])

def process_cancer(chunk):
    cg = chunk["Hybridization REF"]
    chunk = chunk[chunk.columns[1::4]]  # isolate beta values for each participant
    chunk["Hybridization REF"] = cg
    chunk = chunk.set_index("Hybridization REF")
    if "Composite Element REF" in chunk.index:
        chunk = chunk.drop("Composite Element REF", axis=0)
    return chunk.apply(pd.to_numeric).round()

def process_control(chunk):
    # for control, isolate the control samples.
    controlCG = chunk["ID_REF"]
    controlCols = ["1B","2A","3B","3","5","13","19","10","12","14","2","239","200","181","195","308","310","8B","10A","11B"]
    control = chunk[controlCols]
    control["ID_REF"] = controlCG
    return control.set_index("ID_REF").round()

def importCancer():
    cancerRows = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=4500)):
        print("Cancer ",i)
        chunk = process_cancer(chunk)
        targetIndices = cpgIndices.intersection(chunk.index)
        cancerRows = cancerRows.append(chunk.loc[targetIndices])
    cancerRows.loc["Cancer"] = [1] * cancerRows.shape[1]
    cancerRows.loc["Not Cancer"] = [0] * cancerRows.shape[1]
    return cancerRows.transpose()

def importControl():
    controlRows = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=4500,delimiter=r"\t+")):
        print("Control", i)
        chunk = process_control(chunk)
        targetIndices = cpgIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importData():
    control = importControl()
    cancer = importCancer()
    total_data = control.append(cancer)

    X = total_data.iloc[:,:-2].values
    y = total_data.iloc[:,total_data.shape[1]-2:].values
    return X,y

def create_model():
    m = Sequential()
    m.add(Dense(units=1024, activation="relu",input_shape=(10800,)))
    m.add(Dense(units=2, activation="softmax"))
    m.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    return m

data, labels = importData()
kFold = StratifiedKFold(n_splits=5, shuffle=True)
print("Done importing data")
for trainIndices, testIndices in kFold.split(data,np.zeros(shape=len(labels))):
    X_train,X_test = data[trainIndices], data[testIndices]
    Y_train, Y_test = labels[trainIndices], labels[testIndices]
    model = create_model()
    model.fit(X_train,Y_train,batch_size=32,epochs=2)
    score = model.evaluate(X_test,Y_test,batch_size=32)
    print("Score: ", score)
