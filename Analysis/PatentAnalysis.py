#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import LabelBinarizer, StandardScaler, maxabs_scale
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parallel_pandas import ParallelPandas
from scipy import stats
from scipy.stats import gamma
#Probably want to add a year scaling function to the number of bins (This was not done and probably will never be done)
def graph_patents(df, patentnumber):
    cdmapatent = df.loc[df['Pub'] == patentnumber]
    graphlist = cdmapatent['Cited_Date'].apply(sanitize).to_list()
    #print(graphlist)
    n,bins,patches = plt.hist(graphlist,bins = 15, width = 0.75)
    plt.title("Patent Citations for US Patent " +  patentnumber)
    plt.xlabel("Year")
    plt.ylabel("Number of Citations")
    plt.xlim([min(graphlist)+1,max(graphlist)+1])
    #plt.savefig(patentnumber +".png")
    plt.show()
def graph_gamma(df, patentnumber):
    cdmapatent = df.loc[df['Pub'] == patentnumber]
    graphlist = cdmapatent['Cited_Date'].apply(sanitize).to_list()
    #print(graphlist)
    graphlist = sorted(graphlist)
    a = 1
    pdf_gamma = stats.gamma.pdf(graphlist, a, loc = 1990, scale = 0.5)
    plt.plot(graphlist, pdf_gamma)
    plt.title("Gamma Distribution of patent Citations for US Patent " + patentnumber)
    plt.xlabel("Year")
    plt.ylabel("Density")
    plt.show()
#Cutoff refers to the the amount of time it took for a given patent to receive X citations
def sanitize(date):
    date = int(str(date)[:4])
    return date
def find_time(df, patentnumber, cutoff):
    patent = df.loc[df['Pub'] == patentnumber]
    datelist = patent['Cited_Date'].to_list()
    datelist = sorted(datelist)
    datelist = datelist[:cutoff]
    newint = sanitize(datelist[cutoff-1])-sanitize(datelist[0])
    return newint
def find_time_percentile(df, patentnumber, percentile):
    
    patent = df.loc[df['Pub'] == patentnumber]
    datelist = patent['Cited_Date'].to_list()
    datelist = sorted(datelist)
    range = round(len(datelist) * percentile * 0.01)
    newint = sanitize(datelist[range])-sanitize(datelist[0])
    return newint
def quantileranges(percentile, year1, year2):
    df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results"+year1+"-"+year2+".csv")
    counts = df['Pub'].value_counts().to_frame()
    print(counts[:10])
    #counts  = counts.where(counts.gt(counts.quantile(percentile))).stack().sort_index()
    #counts.to_csv("patent_results" + year1+"-"+year2+ str(percentile)+"percentile.csv")

def find_last(df, patentnumber,lastnumber):
    patent = df.loc[df['Pub'] == patentnumber]
    datelist = patent['Cited_Date'].to_list()
    datelist = sorted(datelist)
    datelist = datelist[-lastnumber:]
    return 2023-sanitize(datelist[0])
   


# In[127]:


ParallelPandas.initialize(n_cpu = 8, split_factor =4, disable_pr_bar = False)
#Set the bucket of data here
df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results_test.csv")
print(df)



# In[9]:


total = pd.DataFrame()
for i in range(1990,2000,2):
    df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results" + str(i) + "-" + str(i+2) + "_NEW" + ".csv")
    counts = df['Pub'].value_counts()
    templist = counts.axes[0]
    templist1 = counts.values
    temp = pd.DataFrame()
    temp['Pub'] = templist
    temp['Counts'] = templist1
    
    temp = temp[:20]
    total = pd.concat([total, temp])
total = total[total.Pub != "Pub"]
print(total)


# In[11]:


total.to_csv("top_patents_list.csv")


# In[24]:


temp = pd.read_csv("top_patents_list_cpc.csv")
total = pd.read_csv("top_patents_list.csv")
total = total.drop_duplicates(subset = ['Pub'])
print(total)
total = total.merge(temp, on = "Pub")
total = total.drop(columns = "Unnamed: 0")
total = total.drop_duplicates(subset = ['Pub'])
print(total)


# In[129]:


#df = df.drop_duplicates(subset = ['Pub','Citedby'])
counts = df['Pub'].value_counts()
#value_counts returns a series in key-value form which is undesirable
#The code below should convert it into a usable form
templist = counts.axes[0]
templist1 = counts.values
total = pd.DataFrame()
total['Pub'] = templist
total['Counts'] = templist1
total = total[total['Counts'] >= 1500]
print(total[:20])


# In[300]:


breakthroughs = total['Pub'].to_list()


# In[307]:


#It might be useful to store this data in a file
#The code here can be customized to find how long it took to reach the Xth percentile of citations
timetobreakthrough = pd.DataFrame(columns = ['Pub','Time'])
for i in breakthroughs:
    temp = []
    temp.append(i)
    temp.append(find_last(df,i,100))
    #temp.append(find_time_percentile(df, i , 90))
    #temp.append(find_time(df, i,1000))
    timetobreakthrough.loc[len(timetobreakthrough)] = temp
print(timetobreakthrough)


# In[308]:


#Differentiation here need to be made for counting until X citations and counting until the Xth percentile of citations
timetobreakthrough.to_csv("1998-2000/time_3_1998-2000.csv")
#timetobreakthrough.to_csv("time1998-2000.csv")


# In[67]:


for i in breakthroughs:
    graph_patents(df, i)


# In[ ]:





# In[160]:


#Trying out some clustering here, this is all set up
def cpc_times_cleanup(startyear, tag):
    cpc_codes = pd.read_csv(str(startyear) + "-" + str(startyear+2) + "/patent_cpc_code" + str(startyear) + "-" + str(startyear+2) + ".csv")
    cpc_codes = cpc_codes.drop_duplicates(subset = ["Pub"])
    #Label Binarization
    labels = cpc_codes['cpc'].to_list()
    lb = LabelBinarizer()
    lb.fit(labels)
    newarr = lb.transform(labels)
    print(len(newarr[0]))
    binarized = []
    count = 0
    for i in newarr:
        #temp = []
        #temp.append(i)

        binarized.append(i.tolist().index(1))
        #count +=1
    print(binarized[0])
    cpc_codes['cpc'] = binarized
    patent_times = pd.read_csv(str(startyear) + "-" + str(startyear+2) + "/time" + tag + str(startyear) + "-" + str(startyear+2) + ".csv")
    patent_times = patent_times.drop("Unnamed: 0", axis = 1)
    unified = patent_times.merge(cpc_codes, on = "Pub")
    unified = unified.drop_duplicates(subset = ["Pub"])
    return unified, lb


# In[5]:


def remove_cpc(input):
    input = input.drop(columns = ['cpc', 'Time'],axis = 1)
    return input


# In[27]:


def get_filename(startyear, tag):
    temp = str(startyear)+"-"+str(startyear+2)+"/"+"time"+tag+str(startyear) + "-" + str(startyear+2)+".csv"
    return temp


# In[6]:


def setup_cluster(year):
    times = pd.read_csv(get_filename(year, ""))
    times_10 = pd.read_csv(get_filename(year, "_1_"))
    times_90 = pd.read_csv(get_filename(year, "_2_"))
    times_100 = pd.read_csv(get_filename(year, "_3_"))
    times = times.drop("Unnamed: 0",axis = 1)
    times_10 = times_10.rename(columns = {"Time":"Time_10"})
    times_90 = times_90.rename(columns = {"Time": "Time_90"})
    times_100 = times_100.rename(columns = {"Time": "Time_100"})
    times_10 = times_10.drop("Unnamed: 0", axis = 1)
    times_90 = times_90.drop("Unnamed: 0", axis = 1)
    times_100 = times_100.drop("Unnamed: 0", axis = 1)
    unified = times.merge(times_10, on = "Pub")
    print(unified)
    unified = unified.merge(times_90, on = "Pub")
    unified = unified.merge(times_100, on = "Pub")
    print(unified)
    unified.to_csv(str(year)+"-"+str(year+2)+"/clustering"+str(year) + "-" + str(year+2)+".csv")
    return unified


# In[2]:


#Actual clustering starts here 
#There is 100% a better way to do this
"""
def setup_cluster(year):
    times_10, labels = cpc_times_cleanup(year, '_1_')
    times_10['Times_10'] = times_10['Time']
    times_10 = remove_cpc(times_10)
    print(times_10)
    times, labels = cpc_times_cleanup(year, '')
    times_90, labels = cpc_times_cleanup(year, '_2_')
    times_90['Times_90'] = times_90['Time']
    times_90 = remove_cpc(times_90) 
    last_100, labels = cpc_times_cleanup(year, '_3_')
    last_100['Last_100'] = last_100['Time']
    last_100 = remove_cpc(last_100)
    clustering_unified = times.merge(times_10, on = "Pub")
    clustering_unified = clustering_unified.merge(times_90, on = "Pub")  
    clustering_unified = clustering_unified.merge(last_100, on = "Pub")
    clustering_backup = clustering_unified
    clustering_unified = clustering_unified.drop(columns = "Pub")
    clustering_backup.to_csv(str(year) + '-' + str(year+2) + "/clustering" + str(year)+"-"+str(year+2) + ".csv")
    return labels
    """
#kmeans = MeanShift()


# In[7]:


lb = setup_cluster(1990)
print(lb)


# In[167]:


def convert_to_cpc(lb, input, length):
    test = np.zeros(length).reshape(1,length)
    test[0][input] = 1
    return lb.inverse_transform(test)


# In[30]:


clustering_unified = pd.read_csv("clustering_master.csv")
clustering_unified = clustering_unified[clustering_unified.Time != "Time"]
clustering_backup = clustering_unified
clustering_unified = clustering_unified.drop(columns = ["Pub", "Unnamed: 0"],axis = 1)
#clustering_unified = StandardScaler().fit_transform(clustering_unified)
#clustering_unified = maxabs_scale(clustering_unified)
clustering_backup = clustering_backup.drop(columns = ["Unnamed: 0"])


# In[33]:


#kmeans = MeanShift()
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = "auto")
label = kmeans.fit_predict(clustering_unified)
label = list(label)
print(label.count(1))


clustering_backup['Cluster'] = label
print(clustering_backup)


# In[41]:


#This cell is for generating a top patents file
total_list = total['Pub'].to_list()
temp = []
pubs = clustering_backup['Pub'].to_list()
clusters = clustering_backup['Cluster'].to_list()
for i in total_list:
    temp.append(clusters[pubs.index(i)])
total['Cluster'] = temp
print(total)
total.to_csv("top_patents.csv")


# In[34]:


#Writing a combination function for the biggest files here
def graph_range(startyear, startindex, clusternumber, clustering):
    count = 0
    df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results" + str(startyear) + "-" + str(startyear+2)+".csv")
    temp = pd.read_csv(str(startyear)+"-"+str(startyear+2)+"/"+"clustering"+str(startyear) + "-" + str(startyear+2) + ".csv")
    pub_codes = clustering['Pub'].to_list()[startindex: startindex+len(temp.index)]
    sliced = label[startindex: startindex+len(temp.index)]

    for i in sliced:
        if i == clusternumber:
            graph_patents(df, pub_codes[count])
            #graph_gamma(df, pub_codes[count])
        count +=1
    return startindex+len(temp.index)


# In[43]:


u_labels = np.unique(label)
fig = plt.figure(figsize = (8,4))
count = 0
#A graphing function of all patents must be done by two year ranges otherwise everything goes boom
#clustering_backup = pd.read_csv("1994-1996/clustering1994-1996.csv")
#df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results1992-1994.csv")
#graph_patents(df, "US-5536637-A")

index = 0
for i in range(1990, 2000, 2):
    print(index)
    index = index + graph_range(i, index, 1, clustering_backup)

"""
for i in label:
    if i == 2:
        pub_code = clustering_backup['Pub'].to_list()[count]
        #print(pub_code)
        #pub_code = temp['Pub'].to_list()[newint]
        #print(pub_code)
        graph_patents(df, pub_code)
        #break
    count +=1

for i in u_labels:
    filtered_label = clustering_unified[label == i]
    print(filtered_label)
    plt.scatter(filtered_label["Time_10"], filtered_label["Time_90"],label = i)
plt.title("Clustered Data comparing CPC classification codes and time taken to reach 1500 citations")
plt.xlabel("Time until 1500 citations (years)")
plt.ylabel("Binarized CPC Classification")
plt.savefig("clustering_1.jpg")
plt.show()
"""


# In[ ]:




