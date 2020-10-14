import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas
from functions import show_picture_plot, arranging_picture_vector_as_picture_array
from matplotlib import use

use('Qt5Agg')
# plt.ion()

faces = fetch_olivetti_faces()
X, y = faces['data'], faces['target']
print(f'Data Shape: {X.shape}')
print(f'Label shape: {y.shape}')

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X[i].reshape(64, 64))

plt.show(block=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=10)

df = pandas.DataFrame(X_train)

print(df.head(5))

# Scaling/normalizing data
scalar = MinMaxScaler()
X_Scaled = scalar.fit_transform(X)

pca = PCA(n_components=256)  # Setting the percente of variance should be .95-.99
pca.fit(X_Scaled)

X_reduced = pca.transform(X_Scaled)
X_recovered = pca.inverse_transform(X_reduced)

fig1 = plt.figure()
fig1.suptitle('PCA compression and recovery')
rows = 1
cols = 2
gs = gridspec.GridSpec(rows, cols)
gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0])
gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])
for i in range(3):
    for j in range(3):
        ax00 = fig1.add_subplot(gs00[i, j])
        pic00 = arranging_picture_vector_as_picture_array(16, X_reduced[i + j])
        ax00.imshow(pic00)

        ax01 = fig1.add_subplot(gs01[i, j])
        pic01 = arranging_picture_vector_as_picture_array(64, X_recovered[i + j])
        ax01.imshow(pic01)

plt.show()
