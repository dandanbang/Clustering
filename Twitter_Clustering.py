
# coding: utf-8

# In[5]:

import json
import time
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# ####Loading the Data

# In[6]:

with open('tweets_1M.json','r') as f:
    tweets = json.load(f)


# ##Part 1. Clustering: the baseline

# ###Clustering with K-Means
# ####Write a script that applies the k-means clustering method to the 100K subset. Measure a typical processing time. Experimentally detect the maximum number of clusters k that can be handled with the implementation of the algorithm you are using.

# In[7]:

tweet_data = tweets[:]
tweet_df = pd.DataFrame(data= tweet_data, columns=['lat','lng'])


# ####Coverting Latitude and Longitude in meters

# In[9]:

lat_converted = tweet_df.lat * 89700
lng_converted = tweet_df.lng * 112700
tweet_df['lat'] = lat_converted
tweet_df['lng'] = lng_converted
len(tweet_df)
tweet_df100k = tweet_df[200000:300000]
tweet_df100k


# In[10]:

#Converting this dataframe to array for later use.
tweet_array100k = np.array(tweet_df100k)


# ###Trying out size of clusters 0 - 23 on k-means

# In[11]:

processingTime = {}
processingTimeList = []
n_clustersn = 3
while n_clustersn < 25:
    k_means = KMeans(init='k-means++', n_clusters= n_clustersn, n_init=10)
    t0 = time.time()
    k_means.fit(tweet_df100k)
    t_batch = time.time() - t0
    print ("cluster size" + str(n_clustersn) + ": " + str(t_batch))
    processingTime = {'cluster size': n_clustersn,'time': t_batch}
    processingTimeList.append(processingTime)
    n_clustersn += 1


# ####A plot that shows the positions of the coordinates of the million tweet

# In[19]:

tweet_df.plot(kind='scatter', x='lat', y='lng', color='Red');


# ###Plotting the time it takes from cluster size 1 - 25

# In[4]:

time_df = pd.DataFrame(data= processingTimeList, columns=['cluster size','time'])
time_df.plot(x = 'cluster size', y = 'time')


# ### Cluster Size 100 

# In[21]:

k_means = KMeans(init='k-means++', n_clusters= 100, n_init=10)
t0 = time.time()
k_means.fit(tweet_df100k)
t_batch = time.time() - t0
t_batch


# ### Cluster Size 300

# In[22]:

k_means = KMeans(init='k-means++', n_clusters= 300, n_init=10)
t0 = time.time()
k_means.fit(tweet_df100k)
t_batch = time.time() - t0
t_batch


# ###Cluster Size 500

# In[22]:

k_means = KMeans(init='k-means++', n_clusters= 500, n_init=10)
t0 = time.time()
k_means.fit(tweet_df)
t_batch = time.time() - t0
t_batch


# ####Write a script that applies the MiniBatch k-means method to the 100K subset. Select an appropriate value of a batch size. Measure and note the gain in computational time. Evaluate the maximum number of clusters k that can be handled with the implementation of the algorithm you are using.

# In[23]:

turn = 1
processingTime = {}
processingTimeList = []
batch_sizen = 30
while turn < 200:
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size= batch_sizen,
                      n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(tweet_df100k)
    t_mini_batch = time.time() - t0
    print ("Batch Size " + str(batch_sizen) + ": " + str(t_mini_batch))
    processingTime = {'Batch Size': batch_sizen,'time': t_mini_batch}
    processingTimeList.append(processingTime)
    batch_sizen += 30
    turn += 1


# In[24]:

processingTimeList
time_df2 = pd.DataFrame(data= processingTimeList, columns=['Batch Size','time'])
time_df2.plot(x = 'Batch Size', y = 'time')


# In[25]:

#Finding out the Optimal Batch
Min_BatchSize = time_df2['Batch Size'][time_df2['time'] == time_df2['time'].min()].values
Min_BatchSize


# In[26]:

#Picking Batch Size 900 to run clusters
clusterSize = 1
processingTime = {}
processingTimeList = []
while clusterSize < 100:
    mbk = MiniBatchKMeans(init='k-means++', n_clusters= clusterSize, batch_size= 900,
                      n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(tweet_df100k)
    t_mini_batch = time.time() - t0
    print ("Cluster Size " + str(clusterSize) + ": " + str(t_mini_batch))
    processingTime = {'Cluster Size': clusterSize,'time': t_mini_batch}
    processingTimeList.append(processingTime)
    clusterSize += 1
    turn += 1


# In[19]:

#Picking Batch Size 900 to run clusters and try 100 clusters
mbk = MiniBatchKMeans(init='k-means++', n_clusters= 100, batch_size= 900,
                  n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(tweet_df100k)
t_mini_batch = time.time() - t0
t_mini_batch


# In[27]:

#Plotting the mini-batch size with growing cluster size
processingTimeList
time_df3 = pd.DataFrame(data= processingTimeList, columns=['Cluster Size','time'])
time_df3.plot(x = 'Cluster Size', y = 'time')


# ####Trying Mini Batch for 200 clusters

# In[147]:

mbk = MiniBatchKMeans(init='k-means++', n_clusters= 200, batch_size= 900,
                  n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(tweet_df100k)
t_mini_batch = time.time() - t0
print (t_mini_batch)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)
ft = (k_means_labels, k_means_cluster_centers, k_means_labels_unique)
print ("labels:\n %s, \n cluster centers:\n %s,\n  unique labels:\n %s" % ft)


# In[152]:

#Visualizing Mini-Batch on 200 clusters
n = 200
fig = plt.figure(figsize=(5, 5))
colors = plt.cm.Spectral(np.linspace(0, 1, n))

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(tweet_array100k[my_members, 0],tweet_array100k[my_members, 1], 'w',
            markerfacecolor=col, marker='o', markersize=6)
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col, markeredgecolor='k', markersize=12)
ax.set_title('KMeans MiniBatch')
ax.set_xticks(())
ax.set_yticks(())
plt.show()


# ####Write a script that applies the DBScan method to the 100K subset. Fix the min number of samples in a cluster as 100. Experimentally explore the influence of the connectivity threshold ε on the number of clusters detected by DBScan.

# In[17]:

#ESP 100, observation is that more dense that clusters criteria, the faster the algorithm runs
t_db = time.time()
db = DBSCAN(eps = 200, min_samples = 100).fit(tweet_df100k)
t_fin_db = time.time() - t_db
print(t_fin_db)

db_labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

db_labels_unique = np.unique(db_labels)

print ("labels:\n %s, \n  unique labels:\n %s" % (db_labels, db_labels_unique))


# In[123]:

fig = plt.figure(figsize=(10, 10))
colors = plt.cm.Spectral(np.linspace(0, 1, len(db_labels_unique)))
for k, col in zip(db_labels_unique, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (db_labels == k)

    xy = tweet_array100k[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = tweet_array100k[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
plt.show()


# Prepare a brief write-up of Part 1 containing the following:
# • Reference time of clustering of 100K samples into k=100 clusters with k-means and mini-batch k-means.
# • Maximum number of clusters k_max that your implementation of k-means and mini- batch k-means can handle. Explain the reasons behind this performance bottleneck.
# • The value of ε (call it ε_100) in DBScan resulting in approximately 100 clusters of a minimum of samples (MinPts =100) and the corresponding processing time.

# ##Part 2. Clustering: scalability

# 2.1. For k-means and MiniBatch k-means algorithms, run the experiments and plot the graphs
# representing the computational time as a function of:
# a) Number of data samples (consider the range of 100 to 100’000) for a fixed k =100.
# b) Number of requested clusters k (consider the range of 2 to the k_max)

# In[24]:

#Trying different data size on k-means
processingTime = {}
processingTimeList = []
n_clustersn =  100
datasize = 100
while datasize < 100000:
    k_means = KMeans(init='k-means++', n_clusters= n_clustersn, n_init=10)
    tweet_df2 = tweet_df[:datasize]
    t0 = time.time()
    k_means.fit(tweet_df2)
    t_batch = time.time() - t0
    print ("datasize " + str(datasize) + ": " + str(t_batch))
    processingTime = {'size': datasize,'time': t_batch}
    processingTimeList.append(processingTime)
    n_clustersn += 1
    datasize += 10000


# In[31]:

#Plotting the result 
processingTimeList
time_df3 = pd.DataFrame(data= processingTimeList, columns=['size','time'])
time_df3.plot(x = 'size', y = 'time')


# In[32]:

#Trying different data size on mini-batch

clusterSize = 100
processingTime = {}
processingTimeList = []
datasize = 100
while datasize < 100000:
    mbk = MiniBatchKMeans(init='k-means++', n_clusters= clusterSize, batch_size= 900,
                      n_init=10, max_no_improvement=10, verbose=0)
    tweet_df3 = tweet_df[:datasize]
    t0 = time.time()
    mbk.fit(tweet_df3)
    t_mini_batch = time.time() - t0
    print ("Data Size" + str(datasize) + ": " + str(t_mini_batch))
    processingTime = {'Data Size': datasize,'time': t_mini_batch}
    processingTimeList.append(processingTime)
    clusterSize += 1
    datasize += 5000


# In[34]:

#Plotting the result
processingTimeList
time_df4 = pd.DataFrame(data= processingTimeList, columns=['Data Size','time'])
time_df4.plot(x = 'Data Size', y = 'time')


# In[36]:

#Kmeans, attempting to try larger cluster size 
processingTimekmax = {}
processingTimeListkmax = []
n_clustersn = 2
while n_clustersn < 400:
    k_means = KMeans(init='k-means++', n_clusters= n_clustersn, n_init=10)
    t0 = time.time()
    k_means.fit(tweet_df100k)
    t_batch = time.time() - t0
    print ("Cluster size " + str(n_clustersn) + ": " + str(t_batch))
    processingTimekmax = {'Cluster Size': n_clustersn,'time': t_batch}
    processingTimeListkmax.append(processingTimekmax)
    n_clustersn += 50


# In[37]:

#Plotting the result
time_df4 = pd.DataFrame(data= processingTimeListkmax, columns=['Cluster Size','time'])
time_df4.plot(x = 'Cluster Size', y = 'time')


# In[34]:

#Mini-Batch, attempting to try larger cluster size 
clusterSize = 2
processingTimeminimax = {}
processingTimeListminimax = []
while clusterSize < 500:
    mbk = MiniBatchKMeans(init='k-means++', n_clusters= clusterSize, batch_size= 900,
                      n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(tweet_df100k)
    t_mini_batch = time.time() - t0
    print ("Cluster Size " + str(clusterSize) + ": " + str(t_mini_batch))
    processingTimeminimax = {'Cluster Size': clusterSize,'time': t_mini_batch}
    processingTimeListminimax.append(processingTimeminimax)
    clusterSize += 50


# In[35]:

#Plotting the results
time_df5 = pd.DataFrame(data= processingTimeListminimax, columns=['Cluster Size','time'])
time_df5.plot(x = 'Cluster Size', y = 'time')


# 2.2. For DBScan algorithm, plot the graphs representing computational time as a function of:
# a) Number of samples (consider the range of 100 to 100’000) for a fixed ε_100, MinPts = 100.

# In[40]:

processingTimedb = {}
processingTimeListdb = []
datasize = 100
while datasize < 100000:
    tweet_df2 = tweet_df[:datasize]
    t_db = time.time()
    db = DBSCAN(eps = 200, min_samples = 100).fit(tweet_df100k)
    t_fin_db = time.time() - t_db
    print ("data size " + str(datasize) + ": " + str(t_fin_db))
    processingTimedb = {'data size': datasize,'time':t_fin_db}
    processingTimeListdb.append(processingTimedb)
    datasize += 10000


# In[41]:

#Plotting the results
processingTimedb
time_df = pd.DataFrame(data= processingTimeListdb, columns=['data size','time'])
time_df.plot(x = 'data size', y = 'time')


# Include the graphs in the write-up of your submission. For each of the 3 methods, extrapolate the graphs and provide an estimated time required for your implementation to detect at least 100 clusters in a dataset of 1 million samples.

# ## Part 3. Clustering: 1 million samples problem

# This part deals with the full dataset of 1 million tweets. Your task is to design a system that can handle spatial clustering of 1M samples.
# Considering the memory limitations and scaling properties of the algorithms studied in Part 2, design the clustering system that can be applied to the full dataset. Consider using a hierarchical approach with two (or more) processing stages, where DBScan is applied to each cluster obtained from a run of mini-batch k-means. By varying the parameters of the algorithms, optimize the processing time required to detect clusters of tweets that correspond to important locations in California. We will consider a location “important” if it is characterized with a cluster’s core of at least 100 samples within a radius of 100 meters.

# Describe your approach to the design of the system in a write-up document and provide the total number of clusters detected. Submit your code as a stand-alone script.

# In[20]:

#Run MiniBatch on 1 million tweets. 
mbk = MiniBatchKMeans(init='k-means++', n_clusters=200, batch_size= 900, n_init=10, max_no_improvement=10, 
                      random_state = 333 , verbose=0)
t0 = time.time()
mbk.fit(tweet_df)
t_mini_batch = time.time() - t0
t_mini_batch

mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)

ft = (mbk_means_labels, mbk_means_cluster_centers, mbk_means_labels_unique)
print ("labels:\n %s, \n cluster centers:\n %s,\n  unique labels:\n %s" % ft)
len(mbk_means_labels)


# In[21]:

#Attaching the labels to the 1 million coordinates and group them in the dataframe
tweet_df['label'] = mbk_means_labels
grouped = tweet_df.groupby('label')


# In[22]:

[grouped.get_group(x) for x in grouped.groups]


# In[26]:

labels_keeper = [] #keep all labels from dbscan 


# In[27]:

#run dbscan on the 100 groups of clusters from mini-batch
processingTimedb = {}
processingTimeListdb = []
cluster_number = 0
total_cluster_numbers = 0
while cluster_number < 100:
    tweet_df_new = pd.DataFrame(data = grouped.get_group(cluster_number), columns=['lat','lng'])
    t_db = time.time()
    db = DBSCAN(eps = 100, min_samples = 100).fit(tweet_df_new)
    t_fin_db = time.time() - t_db
    print ("Cluster Number " + str(cluster_number) + ": " + str(t_fin_db))
    processingTimedb = {'Cluster Number': cluster_number,'time':t_fin_db}
    processingTimeListdb.append(processingTimedb)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels_keeper.append(labels)
    total_cluster_numbers += n_clusters_
    print('Estimated number of clusters: %d' % n_clusters_)
    cluster_number += 1


# In[217]:

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# In[218]:

tweet_df_array = np.array(tweet_df_new)


# In[219]:

# plot one of the cluster produced by dbscan
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = tweet_df_array[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = tweet_df_array[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[28]:

total_cluster_numbers # get number of total clusters produced by dbscan


# In[340]:

tweet_df_1 = pd.DataFrame(data = tweet_df_array , columns=['lat','lng'])


# In[343]:

tweet_df = pd.DataFrame(data= tweet_data)


# In[355]:

tweet_df_2 = tweet_df_1.rename(columns={'lat': 'LAT', 'lng': 'LNG'})


# In[358]:

lat_converted = tweet_df_2.LAT / 89700
lng_converted = tweet_df_2.LNG / 112700
tweet_df_2['LAT'] = lat_converted
tweet_df_2['LNG'] = lng_converted
#convert the distance back so we can do matching with the original data to find the tweets that belong in the clusters


# In[361]:

new_tweet_df = pd.concat([tweet_df, tweet_df_2], axis=1) #combine the two dataframes


# In[296]:

new_tweet_df = new_tweet_df.drop('$lat', 1)
new_tweet_df = new_tweet_df.drop('$lng', 1)


# In[362]:

new_tweet_df


# In[527]:

new_List = []
turn = 0
for list in np.array(new_tweet_df):
    if (turn < len(tweet_df_2)):
        new_List.append(list[6])
        turn += 1


# In[501]:

text_list = []
for list in np.array(tweet_df)[]:
    for number in new_List:
        if(list[1] == number):
            print(list[3])


# In[549]:

def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]


# In[537]:

firstHalf, secondHalf = split_list(np.array(tweet_df)) #splitting the a million data in half to make it run faster


# In[548]:

#now we have all 2xxx tweets in the area!
text_list = []
for number in new_List:
    for list in firstHalf:
        if(number == list[1]):
            print(list[3])
            text_list.append(list[3])


# In[556]:

text_list = []
for number in new_List:
    for list in firstHalf:
        if(number == list[1]):
            print(list[3])
            text_list.append(list[3])


# In[560]:

for number in new_List:
    for list in secondHalf:
        if(number == list[1]):
            print(list[3])
            text_list.append(list[3])


# In[561]:

#removing the duplicates
def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


# In[563]:

result_tweet = remove_duplicates(text_list)


# In[597]:

#importing nltk to conduct some text analysis
import nltk
from nltk import word_tokenize


# In[581]:

tweet_str = ''
for tweet in result_tweet:
    tweet_str += str(tweet) + ', ' 


# In[598]:

#tokenizing the words
def tokenize_text(corpus):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sents = sent_tokenizer.tokenize(corpus) # Split text into sentences
    
    return [nltk.word_tokenize(word) for word in raw_sents]


# In[607]:

tweet_str_tokenized = tokenize_text(tweet_str)


# In[630]:

tweet_str2 = ''


# In[641]:

for tweet in tweet_str_tokenized:
    for tw in tweet:
        print(tw)
        tweet_str2 += tw + ', '


# In[652]:

fdist = nltk.FreqDist([w for w in new_tweet_str2])


# In[651]:

new_tweet_str2 = tweet_str2.split()


# In[662]:

#finding the topp 1000 most used words!
fdist.most_common(1000)

