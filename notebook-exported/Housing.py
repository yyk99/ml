#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import tarfile
import urllib 

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/" 
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" 

def fetch_housing_data( housing_url = HOUSING_URL, housing_path = HOUSING_PATH): 
    os.makedirs(housing_path, exist_ok = True) 
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve( housing_url, tgz_path) 
    housing_tgz = tarfile.open( tgz_path) 
    housing_tgz.extractall( path = housing_path) 
    housing_tgz.close()


# In[2]:


fetch_housing_data()


# In[3]:


import pandas as pd 
def load_housing_data( housing_path = HOUSING_PATH): 
    csv_path = os.path.join( housing_path, "housing.csv") 
    return pd.read_csv( csv_path)


# In[4]:


housing = load_housing_data()
housing.head()


# In[5]:


housing.info()


# In[6]:


housing["ocean_proximity"]. value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt 
housing.hist( bins = 50, figsize =( 20,15)) 
plt.show()


# In[9]:


import numpy as np 

def split_train_test( data, test_ratio): 
    shuffled_indices = np.random.permutation( len( data)) 
    test_set_size = int( len( data) * test_ratio) 
    test_indices = shuffled_indices[:test_set_size] 
    train_indices = shuffled_indices[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


len(train_set)


# In[12]:


len(test_set)


# In[13]:


from zlib import crc32 
def test_set_check( identifier, test_ratio): 
    return crc32( np.int64( identifier)) & 0xffffffff < test_ratio * 2** 32 

def split_train_test_by_id( data, test_ratio, id_column): 
    ids = data[id_column] 
    in_test_set = ids.apply( lambda id_: test_set_check( id_, test_ratio)) 
    return data.loc[~in_test_set], data.loc[ in_test_set]


# In[14]:


# Unfortunately, the housing dataset does not have an identifier column. 
# The simplest solution is to use the row index as the ID: 

housing_with_id = housing.reset_index() # adds an ` index ` column
train_set, test_set = split_train_test_by_id( housing_with_id, 0.2, "index")


# In[15]:


#
#
#

housing["income_cat"] = pd.cut(housing["median_income"], 
                                bins =[ 0., 1.5, 3.0, 4.5, 6., np.inf], 
                                labels =[ 1, 2, 3, 4, 5])
housing["income_cat"].hist()


# In[16]:


# Now you are ready to do stratified sampling based on the income category. 
# For this you can use Scikit-Learn’s StratifiedShuffleSplit class:

from sklearn.model_selection import StratifiedShuffleSplit 

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42) 
for train_index, test_index in split.split(housing, housing["income_cat"]): 
    strat_train_set = housing.loc[train_index] 
    strat_test_set = housing.loc[test_index]


# In[17]:


# Let’s see if this worked as expected. 
# You can start by looking at the income category proportions in the test set: 

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[18]:


# Picture 2.10 .... no idea how it was created


# In[19]:


# Now you should remove the income_cat attribute so the data is back to its original state: 

for set_ in (strat_train_set, strat_test_set): 
    set_.drop("income_cat", axis = 1, inplace = True)


# In[20]:


# make a copy to protect original data
housing = strat_train_set.copy()


# ## Visualizing Geographical Data 
# Since there is geographical information (latitude and longitude), 
# it is a good idea to create a scatterplot of all districts to visualize the data (Figure   2-11): 

# In[21]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude")


# In[22]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)


# In[23]:


# Now let’s look at the housing prices (Figure   2-13). 
# The radius of each circle represents the district’s population (option s), 
# and the color represents the price (option c). Wewill use a predefined color map (option cmap) called jet, which ranges 
# from blue (low values) to red (high prices): 16 

housing.plot(kind ="scatter", x = "longitude", y = "latitude", alpha = 0.4, 
             s = housing["population"] / 100, label = "population", figsize =( 10,7), 
             c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, ) 
plt.legend()


# In[24]:


# Looking for Correlations Since the dataset is not too large, 
# you can easily compute the standard correlation coefficient 
# (also called Pearson’s r) between every pair of attributes using the corr() method: 

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[25]:


# Another way to check for correlation between attributes is to use the 
# pandas scatter_matrix() function, which plots every numerical attribute 
# against every other numerical attribute. Since there are now 11 numerical attributes, 
# you would get 112 = 121 plots, which would not fit on a page — so let’s just focus 
# on a few promising attributes that seem most correlated with the median 
# housing value (Figure   2-15): 

from pandas.plotting import scatter_matrix 
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"] 
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[26]:


# The most promising attribute to predict the median house value is the median income, 
# so let’s zoom in on their correlation scatterplot (Figure   2-16): 

housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# ## Experimenting with Attribute Combinations
# Hopefully the previous sections gave you an idea of a few ways you can explore the data and gain insights. You identified a few data quirks that you may want to clean up before feeding the data to a Machine Learning algorithm, and you found interesting correlations between attributes, in particular with the target attribute. You also noticed that some attributes have a tail-heavy distribution, so you may want to transform them (e.g., by computing their logarithm). Of course, your mileage will vary considerably with each project, but the general ideas are similar. 
# 
# One last thing you may want to do before preparing the data for Machine Learning algorithms is to try out various attribute combinations. For example, the total number of rooms in a district is not very useful if you don’t know how many households there are. What you really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful: you probably want to compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at. Let’s create these new attributes:
# 

# In[27]:


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"] 
housing["bedrooms_per_room"] = housing["total_bedrooms"]/ housing["total_rooms"] 
housing["population_per_household"] = housing["population"]/ housing["households"] 

# And now let’s look at the correlation matrix again:

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# ## Prepare the Data for Machine Learning Algorithms
# 
# But first let’s revert to a clean training set (by copying strat_train_set once again). Let’s also separate the predictors and the labels, since we don’t necessarily want to apply the same transformations to the predictors and the target values (note that drop() creates a copy of the data and does not affect strat_train_set): 

# In[28]:


housing = strat_train_set.drop("median_house_value", axis = 1) 
housing_labels = strat_train_set["median_house_value"].copy()


# ### Data Cleaning
# 
# Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions to take care of them. We saw earlier that the total_bedrooms attribute has some missing values, so let’s fix this. You have three options: 
# - Get rid of the corresponding districts. 
# - Get rid of the whole attribute. 
# - Set the values to some value (zero, the mean, the median, etc.).
# 
# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna() methods:

# In[29]:


housing.dropna(subset = ["total_bedrooms"]) # option 1 


# In[30]:


housing.drop("total_bedrooms", axis = 1) # option 2


# In[31]:


median = housing["total_bedrooms"].median() # option 3 
housing["total_bedrooms"].fillna(median, inplace = True)


# If you choose option 3, you should compute the median value on the training set and use it to fill the missing values in the training set. Don’t forget to save the median value that you have computed. You will need it later to replace missing values in the test set when you want to evaluate your system, and also once the system goes live to replace missing values in new data. Scikit-Learn provides a handy class to take care of missing values: *SimpleImputer*. 
# 
# Here is how to use it. First, you need to create a SimpleImputer instance, specifying that you want to replace each attribute’s missing values with the median of that attribute:

# In[32]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy = "median")


# Since the median can only be computed on numerical attributes, you need to create a copy of the data without the text attribute ocean_proximity: 

# In[33]:


housing_num = housing.drop("ocean_proximity", axis = 1)


#  Now you can fit the imputer instance to the training data using the fit() method:

# In[34]:


imputer.fit(housing_num)


# In[ ]:




