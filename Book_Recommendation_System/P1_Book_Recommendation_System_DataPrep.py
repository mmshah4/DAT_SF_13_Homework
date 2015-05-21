
# coding: utf-8

## Exploring Methods to improve Book Recommendations

# #Project Goal:
# 
# #### Build a recommender engine for Books using Book-Crossing datasets and apply variations of Collaborative filtering and Content
# #### filtering to understand pros/cons of each of them. Content based is used mainly for the cold start problem.
# 
# #Data:
# 
# #### Book Crossing data available on http://www2.informatik.uni-freiburg.de/~cziegler/BX/
# #### In this project, Amazon product API and website are used to get several data fields as well. 
# 
# #Tools: 
# #### IPythonNotebook(Anaconda) and Spyder
# 
# #Packages:
# #### •Basic python packages for DS –Numpy, Pandas, Scikit-learn
# #### •Plotting –Matpotlib, PyPlot, Seaborn
# #### •Clustering –kmodes
# #### •Image processing –Scikit-Image
# #### •Recommender systems –python-recsys, divisi2, csc-pysparse, networkx
# #### •API calls –python-amazon-product
# #### •Misc. –random, math, datetime, urllib
# 

# In[2]:

#Import libraries and datasets
import pandas as pd
import numpy as np
get_ipython().magic(u'cd "C:\\Users\\Disha\\Documents\\General_Assembley\\DAT_SF_13\\DAT_SF_13_Homework\\eLibrary\\BX-CSV-Dump"')
book_rating = pd.read_csv("BX-Book-Ratings.csv", sep=";")
users = pd.read_csv("BX-Users.csv", sep=";")
books = pd.read_csv("book2/BX-Books.csv", sep=';', skiprows = [6451, 43666, 51750, 92037, 104318, 121767, 144057, 150788, 157127, 180188, 
                                                               185737, 209388, 209387, 220625, 227932, 228956, 245932, 251295, 259940,
                                                               261528])


# In[51]:

#Collecting basic statistics
#book_rating.describe()
#users.describe()
#books.describe()


# In[3]:

#downsampling for books with highest number of reviews - 5000 books from 271K books
book_rating_down = book_rating.ISBN.value_counts().head(5000)
df_book_rating_down = pd.DataFrame(book_rating_down, columns=['Count'])
df_book_rating_down.reset_index(inplace=True)
df_book_rating_down['ISBN'] = df_book_rating_down['index']
df_book_rating_down['rank'] = df_book_rating_down.index


# In[18]:

df_book_rating_down.head()


# In[4]:

#merging downsampled dataset with book data
book_downsampled = pd.merge(books, df_book_rating_down, how='inner', on=['ISBN'])
book_downsampled.head(10)


# In[ ]:

#Getting Category or Genre of books from Amazon Product API
from amazonproduct import API
api = API(locale='us')
category=[]
for book in np_titles:
    asins = []
    try:
        items = api.item_search('Books', Title=book,Sort='salesrank')
        for item in items:
            asin = item.ASIN
            asins.append(asin)
        result = api.item_lookup(asins[0].text,ResponseGroup = 'BrowseNodes')
        category.append(result.Items.Item.BrowseNodes.BrowseNode.Name)
    except:
        category.append('API Error')
        
#writing the category list into a text file

# Open a file
fo = open("category.txt", "w")
# Write sequence of lines at the end of the file.
for item in category:
  fo.write("%s\n" % item)
# Close opend file
fo.close()


# In[114]:

#importing categories of each book and merging with the dataset
f=open("C:\Users\Disha\Documents\General_Assembley\DAT_SF_13\DAT_SF_13_Homework\eLibrary\BX-CSV-Dump\category.txt")
category = f.read().split('\n')
pd_category = pd.Series(category,name='Category')
new1 = pd_category.reset_index()
new2 = new1.drop(new1.index[[4874]])
book_downsampled2 = pd.merge(book_downsampled, new2,how ='left', on=None,left_index=True, right_index=True)
book_downsampled2.head()
titles = book_downsampled2[['rank','Book-Title','Category','Book-Author','Year-Of-Publication','Publisher']]
titles.head()
titles.to_csv("titles.dat",sep=":",header=False, index=False)


# In[91]:

#plotting a simple bar chart for top 20 categories
import pylab as pl
from matplotlib import pyplot as plt

c_v=book_downsampled2.Category.value_counts().head(20)
OY = c_v.as_matrix()
OX = c_v.index
#import numpy as np
fig = plt.figure(figsize=(12,7.5))
width = 0.5
ind = np.arange(len(OY))
plt.bar(ind, OY)
plt.xticks(ind+width/2, OX, rotation=45)

plt.show()


# In[12]:

#Creating clusters on books dataset
import numpy as np
from kmodes import kmodes

titles = book_downsampled2[['rank','Book-Title','Category','Book-Author','Year-Of-Publication','Publisher']]
titles.head()
titles.to_csv("titles.dat",sep=":",header=False, index=False)
# random categorical data 
titles_clus_data = book_downsampled2[['Category','Book-Author','Year-Of-Publication','Publisher']]
data = titles_clus_data.as_matrix()
type(data)

km = kmodes.KModes(n_clusters=100, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(data)
data[1]


# In[14]:

data = titles_clus_data.as_matrix()


# In[6]:

#testing out sample download 

np_titles = book_downsampled['Book-Title'].as_matrix()
np_images = book_downsampled['Image-URL-L'].as_matrix()


# In[57]:

#Downloading large images from Amazon.com
import urllib
for x in range(0,len(np_images_test)):
    resource = urllib.urlopen(np_images[x])
    filename=np_images[x].split('/')[-1]
    output = open(filename,"wb")
    output.write(resource.read())
    output.close()


# In[10]:

#Processing images to extract color features
import urllib
from PIL import Image
import glob, os
get_ipython().magic(u'cd "C:\\Users\\Disha\\Documents\\General_Assembley\\DAT_SF_13\\DAT_SF_13_Homework\\eLibrary\\BX-CSV-Dump\\images"')
red_mean = []
green_mean = []
blue_mean = []
red_std = []
green_std = []
blue_std = []
for x in range(0,len(np_images)):
    try:
        filename=np_images[x].split('/')[-1]
        im = Image.open(filename)
        out = im.resize((256, 256))
        pix = out.load()
        width, height = 256, 256
        pixel_values = list(out.getdata())
        pixel_values = np.array(pixel_values).reshape((width, height, 3))
        red = np.reshape(pixel_values[:,:,0],65536)
        green = np.reshape(pixel_values[:,:,1],65536)
        blue = np.reshape(pixel_values[:,:,2],65536)
        red_mean.append(np.mean(red))
        red_std.append(np.std(red))
        green_mean.append(np.mean(green))
        green_std.append(np.std(green))
        blue_mean.append(np.mean(blue))
        blue_std.append(np.std(blue))
    except:
        red_mean.append(0)
        red_std.append(0)
        green_mean.append(0)
        green_std.append(0)
        blue_mean.append(0)
        blue_std.append(0)


# In[11]:

#combining image features and creating a dataframe from it
image_stats = np.vstack((red_mean, red_std, green_mean, green_std, blue_mean, blue_std),)
image_stats_t = np.transpose(image_stats)
pd_image_stats = pd.DataFrame(image_stats_t, columns = ['red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std'])
pd_image_stats.head()


# In[116]:

book_downsampled_tmp=pd.read_csv("book_downsampled2.dat", sep=":", header=False)

#book_downsampled2 = pd.merge(book_downsampled, pd_image_stats,how ='left', on=None,left_index=True, right_index=True)
#book_downsampled2.head(20)
#book_downsampled2.to_csv("book_downsampled2.dat", sep=":", header=True)

book_downsampled_tmp['sum_colors'] = book_downsampled_tmp.red_mean + book_downsampled_tmp.red_std+book_downsampled_tmp.green_mean+book_downsampled_tmp.green_std+book_downsampled_tmp.blue_mean+book_downsampled_tmp.blue_std
book_colors_summary = book_downsampled_tmp[['rank','sum_colors']]
book_cate_summary = book_downsampled2[['rank','Category']]
book_color_category = pd.merge(book_colors_summary, book_cate_summary, how='inner', on=['rank'])
book_color_category.head(2)
#book_downsampled2.head(20)


# In[110]:

#Book Clusters
book_downsampled2=pd.read_csv("book_downsampled2.dat", sep=":", header=False)
np_book_dspl2 = book_downsampled2.as_matrix()
len(np_book_dspl2)
clusters_book = np.vstack(clusters)
np_book_dspl_final = np.concatenate((np_book_dspl2[:,11:12],clusters_book),axis=1)
np.savetxt("book_clusters_2.dat",np_book_dspl_final, delimiter =":")


# In[113]:

book_downsampled2.head(2)


# In[170]:

#preparing datasets to see how categories influence ratings
book_ratings_2 = pd.merge(book_rating, df_book_rating_down, how='inner', on=['ISBN'])
book_ratings_2 = book_ratings_2.drop(['index','Count','ISBN'],axis=1,inplace=False)
book_ratings_2 = pd.merge(book_ratings_2, book_color_category, how='inner', on=['rank'])
book_rating_by_categorty = book_ratings_2[['Category','Book-Rating']].groupby('Category').aggregate([np.mean]).sort([('Book-Rating','mean')])
book_rating_by_categorty.tail(10)


# In[172]:

#plotting a simple bar chart for top 20 categories
import pylab as pl
from matplotlib import pyplot as plt
book_rating_by_categorty=book_rating_by_categorty.tail(20)
OY = book_rating_by_categorty.as_matrix()
OX = book_rating_by_categorty.index
#import numpy as np
fig = plt.figure(figsize=(12,7.5))
width = 0.5
ind = np.arange(len(OY))
plt.bar(ind, OY)
plt.xticks(ind+width/2, OX, rotation=45)

plt.show()


# In[126]:

#plotting a scatter plot between colors and ratings
book_ratings_3=book_ratings_2.reindex(np.random.permutation(book_ratings_2.index)).head(10000)
book_ratings_3.head(20)


# In[173]:

#Box plots to see influence of colors
import numpy as np
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')
N = 50
x = book_ratings_3['Book-Rating'].as_matrix()
y = book_ratings_3['sum_colors'].as_matrix()


#build a dataset
positions = [0,1,2,3,4,5,6,7,8,9,10]
score_vector = []
for position in positions:
    score_vector.append(book_ratings_3.sum_colors[(book_ratings_3['Book-Rating'] == position)].values)

from pylab import *
ax = plt.figure()
boxplot(score_vector)
show()


# In[1]:

get_ipython().magic(u'cd "C:\\Users\\Disha\\Documents\\General_Assembley\\DAT_SF_13\\DAT_SF_13_Homework\\eLibrary\\BX-CSV-Dump"')


# In[35]:

#pd_new.columns=['index','bookid','rating','userid']
#pd_new = pd_new.drop('index',axis=1,Inplace=True)
pd_new2=pd_new[['userid','bookid','rating']]
pd_new2.to_csv("book_ratings_3.dat", sep=':', header=False, index=False)

