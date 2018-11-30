from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense, Dropout
import keras
import numpy as np
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import MiniBatchSparsePCA
from methyl_exp import significantSites

import pandas as pd
from sklearn.model_selection import StratifiedKFold

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"

GSE41169controlcols = ['"GSM1009' + x +'"' for x in ["666","667","668","673","674","677","681","685","686","687","688","689","690","691","692","693","694","695","697","723",
                       "727","729","739","742","743","744","745","746","747","748","749","892","893"]]

GSE63499controlcols = ['"GSM1551' + x +'"' for x in ["110","114","118","122","126","132","136","140","144","148","152","156"]]

GSE64495controlcols = ['"GSM157250' + str(x) +'"' for x in range(3,10)] + ['"GSM15725' + str(x) +'"' for x in range(10,65)]

def resetCpGSites(iter=0, sampleNames=None):
    f = open("SignificantCpG_Variances", 'r')
    if sampleNames is not None:
        f.close()
        significantSites(iter,sampleSites=sampleNames)
        f = open("SignificantCpG_Variances_iter"+str(iter),'r')
    cpgSites = f.read()
    cpgSites = cpgSites.split("\n")
    cpgIndices = pd.Index(cpgSites[:-1])
    cpgQuoteIndices = pd.Index([cpg for cpg in cpgSites[:-1]])
    return cpgIndices, cpgQuoteIndices

cpgIndices,cpgQuoteIndices = resetCpGSites()

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

def importCancer(cpgInd):
    cancerRows = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=4500)):
        print("Cancer ",i)
        chunk = process_cancer(chunk)
        targetIndices = cpgInd.intersection(chunk.index)
        cancerRows = cancerRows.append(chunk.loc[targetIndices])
    cancerRows.loc["Cancer"] = [1] * cancerRows.shape[1]
    cancerRows.loc["Not Cancer"] = [0] * cancerRows.shape[1]
    return cancerRows.transpose()

def importControl1(cpgInd): ## Crohn's control
    controlRows = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=4500,delimiter=r"\t+")):
        print("Control 1 ", i)
        chunk = process_control_1(chunk)
        targetIndices = cpgInd.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl2(cpgQuoteInd): # dutch study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE41169_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 2", i)
        chunk = process_control_2(chunk, GSE41169controlcols)
        targetIndices = cpgQuoteInd.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl3(cpgInd): # smoker study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE53045_matrix_processed_GEO.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 3 ", i)
        chunk = process_control_3(chunk)
        targetIndices = cpgInd.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl4(cpgQuoteInd): # folic acid study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE63499_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 4 ", i)
        chunk = process_control_2(chunk, GSE63499controlcols)
        targetIndices = cpgQuoteInd.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importControl5(cpgQuoteInd): # developmental study
    controlRows = pd.DataFrame()
    for i, chunk in enumerate(
            pd.read_table(direc + "GSE64495_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print("Control 5 ", i)
        chunk = process_control_2(chunk, GSE64495controlcols)
        targetIndices = cpgQuoteInd.intersection(chunk.index)
        controlRows = controlRows.append(chunk.loc[targetIndices])
    controlRows.loc["Cancer"] = [0] * controlRows.shape[1]
    controlRows.loc["Not Cancer"] = [1] * controlRows.shape[1]
    return controlRows.transpose()

def importData(cpgInd, cpgQuoteInd):
    control1 = importControl1(cpgInd)
    control2 = importControl2(cpgInd)
    control3 = importControl3(cpgQuoteInd)
    control4 = importControl4(cpgQuoteInd)
    control5 = importControl5(cpgQuoteInd)
    cancer = importCancer(cpgInd)
    total_data = control1.append(control2).append(control3).append(control4).append(control5).append(cancer)
    y = total_data.iloc[:,:2].values
    X = total_data.iloc[:,2:].values
    return X,y, list(total_data.index)

def create_nn_model():
    m = Sequential()
    m.add(Dense(units=1024, activation="relu",input_shape=(216,)))
    m.add(Dense(units=512, activation="relu"))
    m.add(Dense(units=512, activation="relu"))
    m.add(Dense(units=2, activation="softmax"))
    m.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    return m

def create_svm():
    clf = SVC(gamma='auto', probability=True)
    return clf

data, labels, sampleNames = importData(cpgIndices, cpgQuoteIndices)
kFold = StratifiedKFold(n_splits=5, shuffle=True)
i=0

for trainIndices, testIndices in kFold.split(data,np.zeros(shape=len(labels))):
    print("Iteration " + str(i))
    trainingNames, testNames = [sampleNames[i] for i in trainIndices], [sampleNames[i] for i in testIndices]
    cpgIndices,cpgQuoteIndices = resetCpGSites(i,trainingNames)
    data, labels = importData(cpgIndices,cpgQuoteIndices)
    methyl_PCA = PCA()
    results = methyl_PCA.fit_transform(data)
    clusterer = KMeans(n_clusters=2)
    cluster_labels = clusterer.fit_predict(results)

    colors = {0: "r", 1: "g"}
    p = plt.scatter(results[:, 0], results[:, 1], c=[colors[c] for c in cluster_labels])
    plt.title("PCA on Expression Data")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
    colors_label = [0] * 190 + [1] * (len(cluster_labels) - 190)
    p1 = plt.scatter(results[:, 0], results[:, 1], c=[colors[c] for c in colors_label])
    plt.title("PCA on CpG Features Red=Control")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    print("Done importing data")
    i = 0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    X_train,X_test = data[trainIndices], data[testIndices]
    Y_train, Y_test = labels[trainIndices], labels[testIndices]
    model = create_nn_model()
    model.fit(X_train,Y_train,batch_size=32,epochs=3)
    score = model.evaluate(X_test, Y_test, batch_size=32)
    print("Score: ", score)
    i += 1
    # model = create_svm()
    # Y_train = Y_train[:,0]
    # Y_test = Y_test[:,0]
    # model.fit(X_train,Y_train)
    # y_score = model.predict_proba(X_test)
    # print(model.score(X_test,Y_test))

#     fpr, tpr, thresholds = roc_curve(Y_test, y_score[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#     i += 1
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Chance', alpha=.8)
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
