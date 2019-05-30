import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.plotting._converter as pandacnv
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

#plt.style.use('seaborn')
pandacnv.register()

result = pd.read_excel('data.xlsx')

X = result.values

result2 = pd.read_excel('test.xlsx')


X2 = result2.values


# f1 = result['Column1'].values
# f2 = result['MT_002'].values
# X = np.array(list(zip(f1, f2)))
# plt.scatter(f1, f2, c='black', s=7)
# plt.show()

train = X
test = X2


"""
#CODE FOR ELBOW METHOD

# # Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 6))

for k in list_k:
    km = MiniBatchKMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()
"""


 #KMeans
km = MiniBatchKMeans(n_clusters=2)
km.fit(train)
#km.predict(X)
labels = km.predict(train)

print(labels)
"""
plt.scatter(X[:,3],X[:,4], c=km.labels_, cmap='rainbow')
plt.scatter(km.cluster_centers_[:,3] ,km.cluster_centers_[:,4], color='black')
plt.xlabel('Day', fontsize=18)
plt.ylabel('15 Minute Interval', fontsize=16)
plt.show()
"""
sil = silhouette_score(X, km.labels_, sample_size=1000)
print('silhouette score', sil)

sse = km.inertia_
print('SSE', sse)

nmi = normalized_mutual_info_score(km.labels_, labels)
print('NMI', nmi)

ars = adjusted_rand_score(km.labels_, labels)
print("RI", ars)




"""
#CODE FOR SİLHOUETTE ANALYSIS

sample_test = pd.read_excel('test.xlsx')
T = sample_test.values
#
prediction = km.predict(T)
#
"""
"""
sil_score = silhouette_score(X, labels, sample_size=1000)
print(sil_score)

"""

