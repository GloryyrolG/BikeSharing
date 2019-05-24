import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':
    df = pd.read_csv('JC-201903-citibike-tripdata.csv')
    st = pd.DataFrame(df[df['tripduration'] <= 3600],\
        columns = ['start station name', 'start station id', 'start station latitude', 'start station longitude'])
    st = st.drop_duplicates()
    st.index = range(0, st.shape[0])
    st.columns = ['name', 'id', 'lat', 'long']
    ed = pd.DataFrame(df[df['tripduration'] <= 3600],\
        columns = ['end station name', 'end station id', 'end station latitude', 'end station longitude'])
    ed = ed.drop_duplicates()
    ed.index = range(0, ed.shape[0])
    ed.columns = ['name', 'id', 'lat', 'long']
    st = pd.concat([st, ed])
    st = st.drop_duplicates()
    st.index = range(0, st.shape[0])
    d_id = dict()
    for i in range(st.shape[0]):
        d_id[st.at[i, 'id']] = i
    X = np.array(st.iloc[: , 2: ])

    ride = pd.DataFrame(df[df['tripduration'] <= 3600], \
        columns = ['start station id', 'end station id'])
    ride.columns = ['st', 'ed']
    ride['cnt'] = 0
    ride = ride.groupby(['st', 'ed'], as_index = False).count()
    h = dict()
    d_out = dict()
    d_in = dict()
    for i in range(ride.shape[0]):
        d = ride.iloc[i, 2]
        for j in range(2):
            h[ride.iloc[i, j]] = h.get(ride.iloc[i, j], 0) + d
        d_out[ride.iloc[i, 0]] = d_out.get(ride.iloc[i, 0], 0) + d
        d_in[ride.iloc[i, 1]] = d_in.get(ride.iloc[i, 1], 0) + d

    X[: , 0] = (X[: , 0] - 40.7) * 1000
    X[: , 1] = (X[: , 1] + 74.1) * 1000
    Z = np.array([h[i] for i in st['id']])
    pop = np.argsort(-Z)
    out_Z = np.array([d_out.get(i, 0) for i in st['id']])
    out_pop = np.argsort(-out_Z)
    in_Z = np.array([d_in.get(i, 0) for i in st['id']])
    in_pop = np.argsort(-in_Z)
    for i in pop:
        print(st.iloc[i, : 2].values, Z[i], out_Z[i], in_Z[i], \
            'out' if (in_Z[i] >= 30 and out_Z[i] >= 2 * in_Z[i])\
                or out_Z[i] - in_Z[i] >= 300\
            else 'in' if (out_Z[i] >= 30 and in_Z[i] >= 2 * out_Z[i])\
                or in_Z[i] - out_Z[i] >= 300\
            else 'both')
#    plt.subplot('121');
    plt.scatter(X[: , 0], X[: , 1], c = Z, s = 20, zorder = 10)
    plt.colorbar()
    for i in range(ride.shape[0]):
        plt.plot(*list(zip(X[d_id[ride.iloc[i, 0]]], X[d_id[ride.iloc[i, 1]]])), 'r', \
            linewidth = ride.iloc[i, 2] / 150, zorder = 0)
    plt.xlabel('Latitude\'')
    plt.ylabel('Longitude\'')
    plt.title('Station Popularity by Degree')

    '''
    kmeansPredicter = KMeans(n_clusters = 4).fit(X)
    category = kmeansPredicter.predict(X)
    col = 'ygbmr'
    for i, c in enumerate(category):
        plt.scatter(X[i, 0], X[i, 1], color = col[c])
    plt.title('Station Popularity by K-means')
    '''
    '''
    n_clusters = 4
    predictResult = AgglomerativeClustering(n_clusters = n_clusters,
        affinity = 'manhattan',
        linkage = 'average').fit_predict(X)

#    plt.subplot('122')
    for i in range(n_clusters):
        subData = X[predictResult == i]
        plt.scatter(subData[: , 0], subData[: , 1], c = col[i])
    plt.title('Station Popularity by AgglomerativeClustering, Manhattan Distance')
    '''
    plt.xlabel('Latitude\'')
    plt.ylabel('Longitude\'')

    plt.show()
