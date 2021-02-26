import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from statistics import mean
from statistics import median
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.plotting import plot_decision_regions

# df = pd.read_csv("PS_2020.10.26_11.22.58.csv")
df = pd.read_csv('PS_2020.10.26_11.22.58.csv', header=None, sep='\n', skiprows=303)
df = df[0].str.split(',', expand=True)

# print(df.head())

# make the histogram - col 13
datfra = df[13]
disctelescope = df[19]
dfnames = df[1].tolist()     # pl_name

# print("dfnames.head() ", dfnames.head())

for i in range(100):
    print(dfnames[i])

# print("\n", datfra.dtypes)
# plt.hist(datfra)  # density=False would make counts
# plt.ylabel('Count')
# plt.xlabel('Method');
# plt.show()

print(datfra.head())

Z = Counter(datfra)
print(Z)
# Z = Counter(disctelescope)
# print(Z)

# print("Radial Velocity - ", datfra.count("Radial\\ Velocity"))
# print("Imaging - ", datfra.count("Imaging"))
# print("Eclipse Timing Variations - ", datfra.count("Eclipse Timing Variations"))
# print("Transit - ", datfra.count("Transit"))
# print("Astrometry - ", datfra.count("Astrometry"))
# print("Disk Kinematics - ", datfra.count("Disk Kinematics"))
# print("Orbital Brightness Modulation - ", datfra.count("Orbital Brightness Modulation"))
# print("Pulsation Timing Variations - ", datfra.count("Pulsation Timing Variations"))
# print("Microlensing - ", datfra.count("Microlensing"))
# print("Transit Timing Variations - ", datfra.count("Transit Timing Variations "))
# print("Pulsar Timing - ", datfra.count("Pulsar Timing"))

count = 0
for i in range(len(datfra)):
    if(datfra[i] == 'Transit' and disctelescope[i] == '0.95 m Kepler Telescope'):
        count += 1
print('count is - ', count)

labels, values = zip(*Z.items())

print('labels - ', labels)
print('values - ',values)

labels = ['RV', 'IMG', 'ETV', 'TST', 'ATY', 'DK', 'OBM', 'PTV', 'ML', 'TTV', 'PT']
values = [1741, 100, 21, 24529, 1, 1, 15, 2, 304, 106, 12]

labels.remove('TST')
values.remove(24529)

indexes = np.arange(len(labels))
width = 1

# plt.bar(indexes, values, width)
# plt.xticks(indexes + width * 0.1, labels)
# plt.xlabel('Discovery Method', fontsize=18)
# plt.ylabel('No. of Discovered Exoplanets', fontsize=14)
# plt.show()


# train on kepler

# read the other file - habitable planets
hab_plnts = []
ct = 0
f = open("habitable_planets_koi.txt", "r")
for x in f.read().splitlines(): # remove trailing \n
    hab_plnts.append(x)

f.close()
print(len(hab_plnts), hab_plnts)

# final_h_p_lst = []
#
# for j in hab_plnts:
#     if j in dfnames:
#         final_h_p_lst.append(j)

f = open("hab_plnt_data.txt", "w")

sy_mnum = df[11].tolist()
pl_orbper = df[34].tolist()
pl_rade = df[42].tolist()
pl_radj = df[46].tolist()
pl_masse = df[50].tolist()
pl_dens = df[86].tolist()
pl_eqt = df[95].tolist()
st_teff = df[155].tolist()
st_rad = df[159].tolist()
st_lum = df[172].tolist()
pl_imppar = df[112].tolist()


kepler_pl_orbper = []
kepler_pl_rade = []
kepler_pl_eqt = []
kepler_st_teff = []
kepler_st_rad = []

# get indexes for habitable planets
index_hab_plnts = []
for k in hab_plnts:
    for j in range(len(dfnames)):
        if k == dfnames[j]:
            index_hab_plnts.append(j)
            break

f.write("Name \t\t No. Moons \t\t Orbital Period  \t\t Earth Radius \t\t Jupiter Radius \t\t Planet Mass \t\t Planet Density \t\t Equilibrium Temperature \t\t Stellar Effective Temperature \t\t Stellar Radius \t\t Stellar Luminosity \t\t Impact Parameter\n")
for ir in range(len(index_hab_plnts)):
    idx = index_hab_plnts[ir]
    wstr = hab_plnts[ir] + '\t\t' + sy_mnum[idx] + '\t\t' + pl_orbper[idx] + '\t\t' + pl_rade[idx] + '\t\t' + pl_radj[idx] + '\t\t' + pl_masse[idx] + '\t\t' + pl_dens[idx] + '\t\t' + pl_eqt[idx] + '\t\t' + st_teff[idx] + '\t\t' + st_rad[idx] + '\t\t' + st_lum[idx] + '\t\t' + pl_imppar[idx] + '\n'
    f.write(wstr)
    # print(st_teff[idx])
    if pl_orbper[idx] == '' or pl_rade[idx] == '' or pl_eqt[idx] == '' or st_teff[idx] == '' or st_rad[idx] == '':
        continue
    else:
        kepler_pl_orbper.append(None if pl_orbper[idx] == '' else float(pl_orbper[idx]))
        kepler_pl_rade.append(None if pl_rade[idx] == '' else float(pl_rade[idx]))
        kepler_pl_eqt.append(None if pl_eqt[idx] == '' else float(pl_eqt[idx]))
        kepler_st_teff.append(None if st_teff[idx] == '' else float(st_teff[idx]))
        kepler_st_rad.append(None if st_rad[idx] == '' else float(st_rad[idx]))

f.close()


# print('Orbital Period  ---- ', mean(kepler_pl_orbper), ' ', median(kepler_pl_orbper))
# print('Earth Radius  ---- ', mean(kepler_pl_rade), ' ', median(kepler_pl_rade))
# print('Equilibrium Temperature  ---- ', mean(kepler_pl_eqt), ' ', median(kepler_pl_eqt))
# print('Stellar Effective Temperature  ---- ', mean(kepler_st_teff), ' ', median(kepler_st_teff))
# print('Stellar Radius  ---- ', mean(kepler_st_rad), ' ', median(kepler_st_rad))
# kepler_pl_orbper = [x for x in kepler_pl_orbper if x != -1]
# kepler_pl_rade = [x for x in kepler_pl_rade if x != -1]
# kepler_pl_eqt = [x for x in kepler_pl_eqt if x != -1]
# kepler_st_teff = [x for x in kepler_st_teff if x != -1]
# kepler_st_rad = [x for x in kepler_st_rad if x != -1]
print(len(kepler_pl_orbper), 'Orbital Period  ---- ', mean(kepler_pl_orbper), ' ', median(kepler_pl_orbper), ' ', np.std(kepler_pl_orbper))
print(len(kepler_pl_rade), 'Earth Radius  ---- ', mean(kepler_pl_rade), ' ', median(kepler_pl_rade), ' ', np.std(kepler_pl_rade))
print(len(kepler_pl_eqt), 'Equilibrium Temperature  ---- ', mean(kepler_pl_eqt), ' ', median(kepler_pl_eqt), ' ', np.std(kepler_pl_eqt))
print(len(kepler_st_teff), 'Stellar Effective Temperature  ---- ', mean(kepler_st_teff), ' ', median(kepler_st_teff), ' ', np.std(kepler_st_teff))
print(len(kepler_st_rad), 'Stellar Radius  ---- ', mean(kepler_st_rad), ' ', median(kepler_st_rad), np.std(kepler_st_rad))

# 52 habitable

# do it for the unhabitable ones
df2 = pd.read_csv('non_habitable_planets_confirmed_detailed_list.csv')
unhab_plnts = df2['kepler_name'].tolist()

unhab_kepler_pl_orbper = []
unhab_kepler_pl_rade = []
unhab_kepler_pl_eqt = []
unhab_kepler_st_teff = []
unhab_kepler_st_rad = []
# get indexes for unhabitable planets
index_unhab_plnts = []
for k in unhab_plnts:
    for j in range(len(dfnames)):
        if k == dfnames[j]:
            index_unhab_plnts.append(j)
            break
# make the unhab lists
for ir in range(len(index_unhab_plnts)):
    idx = index_unhab_plnts[ir]
    if pl_orbper[idx] == '' or pl_rade[idx] == '' or pl_eqt[idx] == '' or st_teff[idx] == '' or st_rad[idx] == '':
        continue
    else:
        unhab_kepler_pl_orbper.append(None if pl_orbper[idx] == '' else float(pl_orbper[idx]))
        unhab_kepler_pl_rade.append(None if pl_rade[idx] == '' else float(pl_rade[idx]))
        unhab_kepler_pl_eqt.append(None if pl_eqt[idx] == '' else float(pl_eqt[idx]))
        unhab_kepler_st_teff.append(None if st_teff[idx] == '' else float(st_teff[idx]))
        unhab_kepler_st_rad.append(None if st_rad[idx] == '' else float(st_rad[idx]))

print(len(unhab_kepler_pl_eqt)) # 1294


# decide the 0's and 1's

X = []
y = []
for aa in range(len(kepler_pl_orbper)):
    X.append([kepler_pl_orbper[aa], kepler_pl_rade[aa], kepler_pl_eqt[aa], kepler_st_teff[aa], kepler_st_rad[aa]])
    y.append(1)

for aa in range(len(unhab_kepler_pl_eqt)):
    X.append([unhab_kepler_pl_orbper[aa], unhab_kepler_pl_rade[aa], unhab_kepler_pl_eqt[aa], unhab_kepler_st_teff[aa], unhab_kepler_st_rad[aa]])
    y.append(0)

# results on our data
# X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=2) # 80-20 split
k = 12
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)
print("Accuracy of model at K=12 is",metrics.accuracy_score(y_test, Pred_y))

# acc = []
# # Will take some time
# for i in range(1, 40):
#     print(i)
#     neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
#     yhat = neigh.predict(X_test)
#     acc.append(metrics.accuracy_score(y_test, yhat))
#
# print(acc)
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='yellow', markersize=10)
# plt.title('Accuracy vs. K Value (on Training Data)')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.show()
# print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)) + 1)    # Same for [1, 7, 8, 12, 13, 14, 15] Pick the median value - 12


# KNN on predict from external
# grab external data [ALL OF IT, exclude training data]

test_X = []
names_X = []

for pti in range(len(dfnames)):
    if dfnames[pti] in hab_plnts or dfnames[pti] in unhab_plnts:
        continue
    elif pl_orbper[pti] == '' or pl_rade[pti] == '' or pl_eqt[pti] == '' or st_teff[pti] == '' or st_rad[pti] == '':
        continue
    else:
        test_X.append([pl_orbper[pti], pl_rade[pti], pl_eqt[pti], st_teff[pti], st_rad[pti]])
        names_X.append(dfnames[pti])

print(len(test_X))          # 1505
Pred_y = neigh.predict(test_X)

print(Pred_y.tolist().count(0), ' ', Pred_y.tolist().count(1))
preds = Pred_y.tolist()
for vc in range(len(preds)):
    if preds[vc]  == 1:
        print(names_X[vc])
# plt.scatter(range(1,1506), Pred_y, marker='o');
# plt.show()

# # boxplot - FAIL
# fig = plt.figure(figsize=(10, 7))
# plt.boxplot([0,1,1,0,1])
# plt.show()

# sil = []
# kmax = 20
#
# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# for k in range(2, kmax+1):
#   kmeans = KMeans(n_clusters = k).fit(test_X)
#   labels = kmeans.labels_
#   sil.append(silhouette_score(test_X, labels, metric = 'euclidean'))
#
# print(sil)
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(2, 21), sil, color='orange', linestyle='dashed',
#          marker='x', markerfacecolor='green', markersize=8)
# plt.title('Silhouette Score vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Silhouette Score')
# plt.show()


model = KMeans(n_clusters=2)
model.fit(test_X)
labels = model.predict(test_X)

# Make a scatter plot of x and y and using labels to define the colors
xx = []
yy = []
for vg in test_X:
    xx.append(int(vg[2]))
    yy.append(int(float(vg[3])))

print(len(xx), 'xx - ', xx)
print(len(yy), 'yy - ', yy)
print(labels)
print(labels.tolist().count(0))
print(labels.tolist().count(1))

colors = []
for i in range(len(labels)):
    if labels[i] == 0:
        colors.append('orange')
    else:
        colors.append('blue')

plt.scatter(xx, yy, c=colors, alpha=0.5)
plt.xlabel('Equilibrium Temperature [K]')
plt.ylabel('Stellar Effective Temperature [K]')
plt.show()

