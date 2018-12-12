#Contents
#1. Histogram
#2. Outliers
#3. Box plots
#4. Summary Stats
#5. Cumilative Dist Function
#6. Effect Size
#7. Relationship between variables
#8. Correlation
# Pearsons Correlation
#10. Spearman's Correlation
#11. Hypothesis Testing
#12.Normal Distribution & Z-Score



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
from scipy.stats import  norm
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats


import os
print(os.listdir("../breast_cancer_data"))

# read data as pandas data frame
data = pd.read_csv("../breast_cancer_data/data.csv")
data = data.drop(['Unnamed: 32','id'],axis = 1)


# quick look to data
print(data.head())
print(data.shape) # (569, 31)
print(data.columns)


#Histogram:

#noidea what is radius mean! it is not working without it
m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()

frequent_malignant_radius_mean = m[0].max()


index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)


most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_radius_mean)

##From this graph you can see that radius mean of malignant tumors are bigger than radius mean of bening tumors mostly.
#The bening distribution (green in graph) is approcimately bell-shaped that is shape of normal distribution (gaussian distribution)


#2. Outliers
#Errors & rare events are called as outliers
#Calculating outliers

data_bening = data[data["diagnosis"] == "B"]
data_malignant = data[data["diagnosis"] == "M"]
desc = data_bening.radius_mean.describe()
#i. Calculate the 25% & 75%
#ii. inter quartile range (IQR) = Q3-Q1

Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1

#iii. Lower bound =  Q1 - 1.5*IQR
#iv. Upper bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
#v. Anything outside this range is called an outlier
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")

print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)

#3. Box plots
#Outliers can be observed even in Boxplots
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
plt.figure(figsize = (12,8))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

#Summary Stats
print("mean: ",data_bening.radius_mean.mean())
print("variance: ",data_bening.radius_mean.var())
print("standart deviation (std): ",data_bening.radius_mean.std())
print("describe method: ",data_bening.radius_mean.describe())


#4. Cumilative Distribution Function
plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Bening',normed = True,cumulative = True)
sorted_data = np.sort(data_bening.radius_mean)
y = np.arange(len(sorted_data))/float(len(sorted_data)-1)
plt.plot(sorted_data,y,color='red')
plt.title('CDF of bening tumor radius mean')
plt.show()
#
#
# #5. Effect size
# #Notes:https://www.statisticssolutions.com/statistical-analyses-effect-size/
# # Refer the above notes.
# #Here we use standard means difference.
# #We also use Pooled variance: https://stats.stackexchange.com/questions/302725/what-does-pooled-variance-actually-mean
mean_diff = data_malignant.radius_mean.mean() - data_bening.radius_mean.mean()
var_bening = data_bening.radius_mean.var()
var_malignant = data_malignant.radius_mean.var()
var_pooled = (len(data_bening)*var_bening +len(data_malignant)*var_malignant ) / float(len(data_bening)+ len(data_malignant))
effect_size = mean_diff/np.sqrt(var_pooled)
print("Effect size: ",effect_size)
#
#
# #6. Relationship between Variables
# #Best way to check for correlation between variables is scatter plot
plt.figure(figsize = (15,10))
#they are positively correlated
sns.jointplot(data.radius_mean,data.area_mean,kind="regg")
plt.show()
#
# #Relationship betweeen more than 2 variables
sns.set(style = "white")
df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()
#
# #7. Correlation
#ranges from -1 to +1 that is from negative to positive correlation
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()

#8. Covariance
#Covariance increases as two vectors become identical
#Covariance is negative if they point in opposite direction
#np.cov(data.radius_mean,data.area_mean)
print("Covariance between radius mean and area mean: ",data.radius_mean.cov(data.area_mean))
print("Covariance between radius mean and fractal dimension se: ",data.radius_mean.cov(data.fractal_dimension_se))

#Pearson Correlation
#
p1 = data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
p2 = data.radius_mean.cov(data.area_mean)/(data.radius_mean.std()*data.area_mean.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ',p2)

#Spearman's Correlation
#Pearson correlation works well if the relationship between variables are linear and variables are roughly normal. But it is not robust, if there are outliers
ranked_data = data.rank()
spearman_corr = ranked_data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)

#Spearman's correlation is little higher than pearson correlation
#If relationship between distributions are non linear, spearman's correlation tends to better estimate the strength of relationship
#Pearson correlation can be affected by outliers. Spearman's correlation is more robust


#Hypothesis Testing
#explaination: https://www.statisticssolutions.com/hypothesis-testing/
statistic, p_value = stats.ttest_rel(data.radius_mean,data.area_mean)
print('p-value: ',p_value)

#Normal Distribution & Z-Score
# parameters of normal distribution
#lets consider for an example the below mean & std.
mu, sigma = 110, 20  # mean and standard deviation
#create random normal variables with above mean & std
s = np.random.normal(mu, sigma, 100000)
print("mean: ", np.mean(s))
print("standart deviation: ", np.std(s))
# visualize with histogram
plt.figure(figsize = (10,7))
plt.hist(s, 100, normed=False)
plt.ylabel("frequency")
plt.xlabel("IQ")
plt.title("Histogram of IQ")
plt.show()

# If above info were IQs of people around the world
# What if I want to know what percentage of people should have an IQ score between 80 and 140?
#This is where we use Z-score
#z = (x - mean)/std
#z1 = (80-110)/20 = -1.5
#z2 = (140-110)/20 = 1.5
#By using the zscore table we get area = 0.42
#We multiply it with 2 because we are checking on both sides of std.
# 0.4332 * 2 = 0.8664
# 86.64 % of people has an IQ between 80 and 140.
#example 2:
#What percentage of people should have an IQ score less than 80?
#z = (110-80)/20 = 1.5
#Lets look at table of z score 0.4332. 43.32% of people has an IQ between 80 and mean(110).
#If we subtract from 50% to 43.32%, we ca n find percentage of people have an IQ score less than 80.
#50-43.32 = 6.68. As a result, 6.68% of people have an IQ score less than 80.