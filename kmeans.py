import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import math


class KmeansClustering:

    def __init__(self, centroids, df):
        self.centroids = centroids
        self.df = df

    def assignment(self):
        for i in self.centroids.keys():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            self.df['distance_from_{}'.format(i)] = (
                np.sqrt( (self.df['x'] - self.centroids[i][0]) ** 2 + (self.df['y'] - self.centroids[i][1]) ** 2)
            )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
        self.df['closest'] = self.df.loc[:, centroid_distance_cols].idxmin(axis=1)
        self.df['closest'] = self.df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        return self.df


    def update(self):
        for i in self.centroids.keys():
            self.centroids[i][0] = np.mean(self.df[self.df['closest'] == i]['x'])
            self.centroids[i][1] = np.mean(self.df[self.df['closest'] == i]['y'])
        return self.centroids


    def wc_ssd(self):
        wc = 0
        for i in self.centroids.keys():
            wc += sum((self.df[self.df['closest']==i]['x']- self.centroids[i][0]) ** 2
            + (self.df[self.df['closest']==i]['y'] - self.centroids[i][1]) ** 2)
        return wc


    def sc(self):
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
        all_si = []
        for i in self.centroids.keys():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            self.df['distance_from_{}'.format(i)] = (
                np.sqrt((self.df['x'] - self.centroids[i][0]) ** 2 + (self.df['y'] - self.centroids[i][1]) ** 2)
            )
            a = float(sum(self.df[self.df['closest'] == i-1][centroid_distance_cols[i-1]]))
            b = float(sum(self.df[self.df['closest'] != i-1][centroid_distance_cols[i-1]]))
            try:
                si = (b-a)/max(a,b)
            except ZeroDivisionError:
                si = 0
            all_si.append(si)
        all_si = [0 if math.isnan(x) else x for x in all_si]
        average_si = sum(all_si)/len(all_si)
        return average_si


def NMI(dataSet):
    hy = entropy_class(dataSet)
    hc = entropy_cluster(dataSet)
    icg = mutual_information(dataSet)
    return (icg)/(hy+hc)


def entropy_class(dataSet):
    event_count = {}
    entropy = 0
    for x in pd.unique(dataSet['d']):
        event = dataSet[dataSet['d'] == x]
        event_count[x] = len(event)
    for e in event_count:
        prob_i = event_count.get(e)
        prob_total = float(len(dataSet))
        entropy += -prob_i / prob_total * math.log(prob_i / prob_total, 2)
    return entropy


def entropy_cluster(dataSet):
    event_count = {}
    entropy = 0
    for x in pd.unique(dataSet['closest']):
        event = dataSet[dataSet['closest'] == x]
        event_count[x] = len(event)
    for e in event_count:
        prob_i = event_count.get(e)
        prob_total = float(len(dataSet))
        entropy += -prob_i / prob_total * math.log(prob_i / prob_total, 2)
    return entropy


def mutual_information(dataSet):
    entropy = 0
    for c in pd.unique(dataSet['closest']):
        cluster = dataSet[dataSet['closest'] == c]
        for g in pd.unique(dataSet['d']):
            event = cluster[cluster['d'] == g]
            cls = dataSet[dataSet['d'] == g]
            prob_total = float(len(dataSet))
            prob_cg = len(event)/prob_total
            prob_c = len(cluster)/prob_total
            prob_g = len(cls)/prob_total
            try:
                entropy += prob_cg * math.log(prob_cg / (prob_c * prob_g), 2)
            except:
                entropy += 0

    return entropy


def fit(kmeans):
    kmeans.assignment()
    itr = 0
    while True:
        if itr == 49:
            break
        itr += 1
        closest_centroids = kmeans.df['closest'].copy(deep=True)
        kmeans.update()
        kmeans.assignment()
        if closest_centroids.equals(kmeans.df['closest']):
            break
    return kmeans


def main():
    com = sys.argv
    dataFilename,  k = com[1], com[2]

    embedding_data = np.genfromtxt(dataFilename, delimiter=',', dtype='uint8')
    df = pd.DataFrame({
        'x': embedding_data[:, 2],
        'y': embedding_data[:, 3],
        'd': embedding_data[:, 1]
    })
    np.random.seed(0)
    idx = np.random.randint(0, len(embedding_data), k)
    # centroids[i] = [x, y]
    centroids = {
        i+1: [embedding_data[idx[i], 2], embedding_data[idx[i], 3]]
        for i in range(k)
    }
    kmeans = KmeansClustering(centroids, df)
    kmeans = fit(kmeans)
    wc = kmeans.wc_ssd()
    sc = kmeans.sc()
    nmi = NMI(kmeans.df)
    print 'WC-SSD:{}'.format(wc)
    print 'SC:{}'.format(sc)
    print 'NMI:{}'.format(nmi)

main()