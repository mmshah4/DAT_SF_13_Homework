
import recsys.algorithm
recsys.algorithm.VERBOSE = True
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

# Read book info
from recsys.datamodel.item import Item
def read_items(filename):
    items = dict()
    for line in open(filename):
        data =  line.strip('\r\n').split(':')
        item_id = int(data[0])
        item_name = data[1]
        category = data[2]
        Book_author = data[3]
        year_of_publication = data[4]
        publisher = data[5]
        item = Item(item_id)
        item.add_data({'Book': item_name, 'category': category,'Book_author':Book_author,
        'year_of_publication':year_of_publication,'publisher':publisher})
        items[item_id] = item
    return items

filename = 'book_ratings_3.dat'

import pandas as pd
import numpy as np
import random

pd_book = pd.read_csv(filename, sep=':', index_col = False, header=None,names = ['rank','user_id','ratings'],dtype=int)
pd_books_clusters = pd.read_csv('book_clusters.dat', sep=':', header=None, names = ['rank','cluster'], dtype=int)
pd_books_clusters[['rank','cluster']] = pd_books_clusters[['rank','cluster']].astype(int)
#pd_books_clusters.head()
pd_book2 = pd.merge(pd_book,pd_books_clusters,on='rank',how='inner')
pd_book2 = pd_book2.reindex(np.random.permutation(pd_book.index))
#pd_book2.fillna(99999999)
pd_book2.dropna(subset = ['rank','user_id','ratings','cluster'])
pd_book2 = pd_book2.dropna(how='all')
ix_missing = pd_book2.isnull()
#ix_missing.head()
p#d_book2.isnull().sum()
#pd_book2.head()
pd_book2.to_csv('book_ratings_4.dat',sep=':', index=False, header=None)

filename2 = 'book_ratings_4.dat'

format = {'col':int(1), 'row':int(3), 'value':int(2),'ids':int}

data = Data()

data.load(filename2, sep=':', format=format)

train, test = data.split_train_test(percent=60)
svd = SVD()
svd.set_data(train)
K=50

svd.compute(k=K, min_values=1, pre_normalize=False, mean_center=True, post_normalize=True)

#import sys

#Evaluation using prediction-based metrics
#test[1]

rmse = RMSE()
mae = MAE()
count = 0
error_count =0
zeros = 0
book_predict = []
user_predict = []
for   rating,item_id, user_id in train.get():
    if rating>0:    
        try:
             pred_rating = svd.predict(item_id, user_id, 0, 10)
             user_predict.append(user_id)
             #print pred_rating, rating, item_id, user_id
             rmse.add(rating, pred_rating)
             mae.add(rating, pred_rating)
             count = count + 1
        except KeyError:
            error_count = error_count+1
            book_predict.append(item_id)
            continue
    else:
        zeros = zeros+1
print count, error_count, zeros
print 'RMSE=%s' % rmse.compute()
print 'MAE=%s' % mae.compute()