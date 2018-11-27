from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense, Dropout
import keras
import numpy as np
from keras.optimizers import Adam
from sklearn.svm import SVC

import pandas as pd
from sklearn.model_selection import StratifiedKFold

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"

GSE41169controlcols = ['"GSM1009' + x +'"' for x in ["666","667","668","673","674","677","681","685","686","687","688","689","690","691","692","693","694","695","697","723",
                       "727","729","739","742","743","744","745","746","747","748","749","892","893"]]

GSE63499controlcols = ['"GSM1551' + x +'"' for x in ["110","114","118","122","126","132","136","140","144","148","152","156"]]

GSE64495controlcols = ['"GSM157250' + str(x) +'"' for x in range(3,10)] + ['"GSM15725' + str(x) +'"' for x in range(10,65)]

rm_quote = lambda x: x.replace('"', '')

f = open("SignificantCpG_Variances",'r')
cpgSites = f.read()
cpgSites = cpgSites.split("\n")
cpgIndices = pd.Index(cpgSites[:-1])
cpgQuoteIndices = pd.Index([cpg for cpg in cpgSites[:-1]])

def process_cancer(chunk):
    cg = chunk["Hybridization REF"]
    chunk = chunk[chunk.columns[1::4]]  # isolate beta values for each participant
    chunk["Hybridization REF"] = cg
    chunk = chunk.set_index("Hybridization REF")
    if "Composite Element REF" in chunk.index:
        chunk = chunk.drop("Composite Element REF", axis=0)
    return chunk.apply(pd.to_numeric).round()

def process_control_1(chunk):
    # for control, isolate the control samples.
    controlCG = chunk["ID_REF"]
    controlCols = ["1B","2A","3B","3","5","13","19","10","12","14","2","239","200","181","195","308","310","8B","10A","11B"]
    control = chunk[controlCols]
    control["ID_REF"] = controlCG
    return control.set_index("ID_REF").round()

def process_control_3(chunk):
    controlCG = chunk["ID_REF"]
    controlCols = ["GSM1280" + str(937+x) for x in range(0,27)]+["GSM1280" + str(991+x) for x in range(0,9)] + ["GSM128" + str(1000+x) for x in range(0,25)]
    control = chunk[controlCols]
    control["ID_REF"] = controlCG
    return control.set_index("ID_REF").round()

def process_control_2(chunk, controlColNames):
    chunk['"ID_REF"'] = [i[1:len(i) - 1] for i in chunk['"ID_REF"']]
    controlCG = chunk['"ID_REF"']
    control = chunk[controlColNames]
    control["ID_REF"] = controlCG
    return control.set_index("ID_REF").apply(pd.to_numeric).round()

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

def importControl1(): ## Crohn's control
    controlRows = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=4500,delimiter=r"\t+")):
        print("Control 1 ", i)
        chunk = process_control_1(chunk)
        targetIndices = cpgIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl2(): # dutch study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE41169_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control", i)
        chunk = process_control_2(chunk, GSE41169controlcols)
        targetIndices = cpgQuoteIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl3(): # smoker study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE53045_matrix_processed_GEO.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 3 ", i)
        chunk = process_control_3(chunk)
        targetIndices = cpgIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl4(): # folic acid study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE63499_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 4 ", i)
        chunk = process_control_2(chunk, GSE63499controlcols)
        targetIndices = cpgIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl5(): # developmental study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE64495_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 5 ", i)
        chunk = process_control_2(chunk, GSE64495controlcols)
        targetIndices = cpgIndices.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importData():
    control1 = importControl1()
    control2 = importControl2()
    control3 = importControl3()
    control4 = importControl4()
    control5 = importControl5()
    cancer = importCancer()
    # cancer = cancer.sample(n=20)
    total_data = control1.append(control2).append(control3).append(control4).append(control5).append(cancer)
    y = total_data.iloc[:,:2].values
    X = total_data.iloc[:,2:].values
    return X,y

def create_nn_model():
    m = Sequential()
    m.add(Dense(units=1024, activation="relu",input_shape=(10800,)))
    m.add(Dense(units=2, activation="softmax"))
    m.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    return m

def create_svm():
    clf = SVC(gamma='auto')
    return clf
data, labels = importData()
kFold = StratifiedKFold(n_splits=5, shuffle=True)
print("Done importing data")
for trainIndices, testIndices in kFold.split(data,np.zeros(shape=len(labels))):
    X_train,X_test = data[trainIndices], data[testIndices]
    Y_train, Y_test = labels[trainIndices], labels[testIndices]
    # model = create_nn_model()
    # model.fit(X_train,Y_train,batch_size=32,epochs=3)
    # score = model.evaluate(X_test, Y_test, batch_size=32)
    # print("Score: ", score)
    model = create_svm()
    Y_train = Y_train[:,0]
    Y_test = Y_test[:,0]
    model.fit(X_train,Y_train)
    print(model.score(X_test,Y_test))

