import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas

faces = fetch_olivetti_faces()
X, y = faces['data'], faces['target']
print(f'Data Shape: {X.shape}')
print(f'Label shape: {y.shape}')

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X[i].reshape(64, 64))

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=10)

df = pandas.DataFrame(X_train)

print(df.head(5))

# Scaling/normalizing data
scalar = MinMaxScaler()
X_Scaled = scalar.fit_transform(X)

pca = PCA(n_components=0.99) #Setting the percente of variance should be .95-.99
pca.fit(X_Scaled)

X_reduced = pca.transform(X_Scaled)
X_recovered = pca.inverse_transform(X_reduced)

print(X_reduced.shape)


# plt.figure()
# plt.subplots(1, 2)
# plt.subplot(1, 1)
#todo add image plot of original and recoveded images

