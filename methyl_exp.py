import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.decomposition
import matplotlib.pyplot as plt

direc = "C:/Users/chsue/Documents/MITyear3/6.047/GDACLAMLmethyl450/data/"

GSE41169controlcols = ['"GSM1009' + x +'"' for x in ["666","667","668","673","674","677","681","685","686","687","688","689","690","691","692","693","694","695","697","723",
                       "727","729","739","742","743","744","745","746","747","748","749","892","893"]]

GSE63499controlcols = ['"GSM1551' + x +'"' for x in ["110","114","118","122","126","132","136","140","144","148","152","156"]]

GSE64495controlcols = ['"GSM157250' + str(x) +'"' for x in range(3,10)] + ['"GSM15725' + str(x) +'"' for x in range(10,65)]

# ---------------- DATA (PANDAS) PROCESSING -------------------------------- #
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
    for chunk1, chunk2,chunk3, chunk4,chunk5,chunk6 in zip(pd.read_table(direc+"LAMLmethyl450.txt", chunksize=4500),
                                           pd.read_table(direc+"GSE32148_matrix_processed_peripheralBlood.txt",chunksize=4500,delimiter=r"\t+"),
                                                           pd.read_table(direc + "GSE41169_series_matrix.txt", chunksize=4500, delimiter=r"\t+"),
                                                           pd.read_table(direc + "GSE53045_matrix_processed_GEO.txt", chunksize=4500, delimiter=r"\t+"),
                                                           pd.read_table(direc + "GSE63499_series_matrix.txt", chunksize=4500, delimiter=r"\t+"),
                                                           pd.read_table(direc + "GSE64495_series_matrix.txt", chunksize=4500, delimiter=r"\t+")):
        print(count)
        cancer_chunk, control_chunk_1,control_chunk_2,control_chunk_3,control_chunk_4,control_chunk_5 = process_cancer(chunk1), \
                                                                                                        process_control_1(chunk2),\
                                                                                                        process_control_2(chunk3,GSE41169controlcols), \
                                                                                                        process_control_3(chunk4), \
                                                                                                        process_control_2(chunk5,GSE63499controlcols), \
                                                                                                        process_control_2(chunk6, GSE64495controlcols)
        total_chunk = pd.concat([cancer_chunk,control_chunk_1,control_chunk_2,control_chunk_3,control_chunk_4,control_chunk_5],axis=1).dropna()
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
