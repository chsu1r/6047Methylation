import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.decomposition
import matplotlib.pyplot as plt

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"
# ---------------- IMPORTING CANCER DATA ------------------------- #
print("Importing cancer type....")  # cancer samples are from TCGA-LAML
methyl = pd.DataFrame()
for i,chunk in enumerate(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=10**3)):
    if i == 10: break  # as of right now, importing everything is a Lot, so import the first 10k CpG sites
    methyl = methyl.append(chunk)
cg = methyl["Hybridization REF"]
methyl = methyl[methyl.columns[1::4]]  # isolate beta values for each participant
methyl["Hybridization REF"] = cg
methyl = methyl.set_index("Hybridization REF").drop("Composite Element REF",axis=0).apply(pd.to_numeric)

# ---------------- IMPORTING CONTROL DATA ------------------------- #
print("importing control....")  # control samples are peripheral blood samples from Crohn's, ulcerative colitis study
control = pd.DataFrame()
for j, chunk in enumerate(pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=10**3,delimiter=r"\t+")):
    if j == 10: break  # import the same first 10k CpG sites
    control = control.append(chunk)

# for control, isolate the control samples.
controlCG = control["ID_REF"]
controlCols = ["1B","2A","3B","3","5","13","19","10","12","14","2","239","200","181","195","308","310","8B","10A","11B"]
control = control[controlCols]
control["ID_REF"] = controlCG
control = control.set_index("ID_REF")
# ---------------- END IMPORTING DATA ------------------------- #

cancerVcontrol = methyl.iloc[:,0:20].join(control).dropna(axis=0)

# --------- calculate the abs difference in averages between cancer/control for CpG sites ------- #
methylAvgs = methyl.iloc[:,:20].mean(axis=1)
controlAvgs = control.mean(axis=1)
diffAvgs = pd.concat([controlAvgs,methylAvgs],axis=1).diff(axis=1).iloc[:,1].dropna().abs()
diffIndices = diffAvgs.nlargest(100).index

methylAvgs = methylAvgs[diffIndices]
controlAvgs = controlAvgs[diffIndices]
p = plt.scatter(range(len(methylAvgs)),methylAvgs,c='r')
plt.scatter(range(len(controlAvgs)), controlAvgs,c='g')
plt.title("Methylation (red-cancer, green-control)")
plt.xlabel("Index of CpG site")
plt.ylabel("Methylation Beta")
plt.show()  # This plot shows the top 100 CpG sites with the highest difference between cancer/noncancer cases.

variances = methyl.var(axis=0)
variances = variances.nlargest(100)
