
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)



# In[44]:


boston.keys()


# In[45]:


boston.data.shape


# In[46]:


print(boston.feature_names)


# In[47]:


print (boston.DESCR)


# In[48]:


bos.head()


# In[49]:


bos.columns = boston.feature_names
bos.head()


# In[50]:


boston.target[:5]


# In[51]:


bos["PRICE"] = boston.target


# In[52]:


bos.head()


# In[53]:


#Skikit learning
from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)
#linear object
lm = LinearRegression()
lm.fit(X,bos.PRICE)


# In[54]:


#Intercept and coefficients
print("Estimated intercept coefficient:",lm.intercept_)


# In[55]:


print("number of coefficients:",len(lm.coef_))


# In[56]:


#column 0 is 'features' and 1 is 'estimated coefficients'
pd.DataFrame(list(zip(X.columns, lm.coef_)))


# In[57]:


#plot between true housing prices and true RM
plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling(RM)")
plt.ylabel("Housing price")
plt.title("Relationship between RM and Price")
plt.show()


# In[58]:


#predicting prices for first 25 houses
lm.predict(X)[0:25]


# In[65]:


#plot between true prices and predicted prices
plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("True Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("True prices vs Predicted prices:")


# In[60]:


#Mean Squared Error
mseFull = np.mean((bos.PRICE - lm.predict(X))**2)
print(mseFull)


# In[61]:


#Train-test split
from sklearn.cross_validation import cross_val_score
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, bos.PRICE, test_size=0.33, random_state= 5)


print("Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train-lm.predict(X_train))**2))
print("Fit a model X_train, and calculate MSE with X_test, Y_test:",np.mean(Y_test - lm.predict(X_test))**2)


# In[62]:


#Residuals vs Residual plot (blue) Training and (green) tezt data
plt.scatter(lm.predict(X_train),lm.predict(X_train) - Y_train,c='b',s=40,alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test) - Y_test,c='g',s=40)
plt.hlines(y = 0, xmin = 0, xmax = 50)
plt.title("Residual plot using training(blue) and test(green) data")
plt.ylabel("Residuals")

