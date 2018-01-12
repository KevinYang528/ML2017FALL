import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys

images = np.load(sys.argv[1])
print('load images done!')

# PCA
pca = PCA(n_components=400, copy=True, whiten=True, 
          svd_solver='full', tol=0.0, iterated_power='auto', random_state=0)

images_pca = pca.fit_transform(images)
print('pca shape: ', images_pca.shape)

# K-means
kmeans = KMeans(n_clusters=2, random_state=0, max_iter=300).fit(images_pca)
print('k-means done!')

# load test_case
a = []
b = []
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:        
        num, a_tmp, b_tmp = line.strip('\n').split(',')
        a.append(int(a_tmp))
        b.append(int(b_tmp))

print('load test_case done!')

# output
count_1 = 0
with open(sys.argv[3], 'w+') as f:
    f.write('ID,Ans\n')
    for i in range(len(a)):        
        if (kmeans.labels_[a[i]] == kmeans.labels_[b[i]]):
            output = str(i) + ',' + '1' + '\n'
            count_1 += 1
        else:
            output = str(i) + ',' + '0' + '\n'
        f.write(output)
    
f.close
print('count:', count_1)