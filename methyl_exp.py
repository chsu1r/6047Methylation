import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.decomposition
import matplotlib.pyplot as plt

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"

# ---------------- DATA (PANDAS) PROCESSING -------------------------------- #
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

# ---------------- IMPORTING CANCER DATA ------------------------- #
print("Importing cancer type....")  # cancer samples are from TCGA-LAML
def import_all_cancer():
    methyl = pd.DataFrame()
    for i,chunk in enumerate(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=10**3)):
        if i == 30: break  # as of right now, importing everything is a Lot, so import the first 10k CpG sites
        methyl = methyl.append(chunk)
    return process_cancer(methyl)

# methyl = import_all_cancer()

# ---------------- IMPORTING CONTROL DATA ------------------------- #
print("importing control....")  # control samples are peripheral blood samples from Crohn's, ulcerative colitis study
def import_all_control():
    control = pd.DataFrame()
    for j, chunk in enumerate(pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=10**3,delimiter=r"\t+")):
        if j == 30: break  # import the same first 10k CpG sites
        control = control.append(chunk)
    return process_control(control)

# control = import_all_control()
# ---------------- END IMPORTING DATA ------------------------------ #

# cancerVcontrol = methyl.iloc[:,0:20].join(control).dropna(axis=0)

# --------- calculate the abs difference in averages between cancer/control for CpG sites ------- #
def identify_significant_cpgSites():
    cpgSites = pd.Index([])
    count = 0
    for chunk1, chunk2 in zip(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=4500),
                                           pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=4500,delimiter=r"\t+")):
        print(count)
        cancer_chunk, control_chunk = process_cancer(chunk1), process_control(chunk2)
        total_chunk = pd.concat([cancer_chunk,control_chunk],axis=1).dropna()
        variances = total_chunk.var(axis=1)
        diffIndices = variances.nlargest(100).index
        cpgSites = cpgSites.union(diffIndices)
        count += 1
    return cpgSites

# methylAvgs = methyl.iloc[:,:20].mean(axis=1)
# controlAvgs = control.mean(axis=1)
# diffAvgs = pd.concat([controlAvgs,methylAvgs],axis=1).diff(axis=1).iloc[:,1].dropna().abs()
# diffIndices = diffAvgs.nlargest(300).index
#
# methylAvgs = methylAvgs[diffIndices]
# controlAvgs = controlAvgs[diffIndices]

significantSites = identify_significant_cpgSites()
f = open("SignificantCpG_Variances", 'w')
for site in significantSites:
    f.write(site + "\n")
f.close()
# --------------- PLOTTING ----------------------------------------#
# p = plt.scatter(range(len(methylAvgs)),methylAvgs,c='r')
# plt.scatter(range(len(controlAvgs)), controlAvgs,c='g')
# plt.title("Methylation (red-cancer, green-control)")
# plt.xlabel("Index of CpG site")
# plt.ylabel("Methylation Beta")
# plt.show()  # This plot shows the top 100 CpG sites with the highest difference between cancer/noncancer cases.
#
#
# # not used yet
# variances = methyl.var(axis=0)
# variances = variances.nlargest(300)
