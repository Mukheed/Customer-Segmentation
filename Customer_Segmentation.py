# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Load the data
data=pd.read_excel("/content/ecom customer_data.xlsx")

# Display the first few rows of the data
data.head()

# Create a copy of the data
df=data.copy()

# Display information about the data
df.info()

# Display summary statistics of the data
df.describe()

# Check for duplicate rows in the data
df[df.duplicated()]

# Check for missing values in the data
df.isna().sum()

# Fill missing values in the 'Gender' column with the mode
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])

# Check for missing values in the data again
df.isna().sum().sum()

# Display the count of each gender
df.Gender.value_counts()

# Plot a countplot of the gender distribution
sns.countplot(data=df,x='Gender')
plt.show()

# Plot a countplot of the orders distribution
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(data=df,x='Orders')

# Plot a countplot of the orders distribution by gender
plt.subplot(1,2,2)
sns.countplot(data=df,x='Orders',hue='Gender')
plt.suptitle("Overall Orders VS Gender wise Orders")
plt.show()

# Define a function to plot boxplots of the data
cols=list(df.columns[2:])
def dist_list(lst):
  plt.figure(figsize=(30,30))
  for i, col in enumerate(lst,1):
    plt.subplot(6,6,i)
    sns.boxplot(data=df,x=df[col])
dist_list(cols)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(20,15))
sns.heatmap(df.iloc[:,3:].corr())
plt.show()

# Plot histograms of the data
df.iloc[:2,:].hist(figsize=(40,30))
plt.show()

# Create a new dataframe with a 'Total Search' column
new_df=df.copy()
new_df['Total Search']=new_df.iloc[:,3:].sum(axis=1)

# Sort the dataframe by 'Total Search' in descending order
new_df.sort_values('Total Search', ascending=False)

# Plot a barplot of the top 10 customers by 'Total Search'
plt.figure(figsize=(13,8))
plt_data=new_df.sort_values('Total Search',ascending=False)[['Cust_ID','Gender','Total Search']].head(10)
sns.barplot(data=plt_data,
            x='Cust_ID',
            y='Total Search',
            hue='Gender',
            order=plt_data.sort_values('Total Search',ascending=False).Cust_ID)
plt.title("Top 10 Cust_ID based on Total Searches")
plt.show()

# Scale the data using MinMaxScaler
x=df.iloc[:,2:].values
scale=MinMaxScaler()
features=scale.fit_transform(x)

# Perform KMeans clustering with different numbers of clusters
inertia=[]
for i in range(1,16):
  k_means=KMeans(n_clusters=i)
  k_means=k_means.fit(features)
  inertia.append(k_means.inertia_)

# Plot the inertia values
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.plot(range(1,16),inertia, 'bo-')
plt.xlabel('No of clusters'),plt.ylabel('Inertia')

# Plot the Elbow Visualizer
plt.subplot(1,2,2)
kmeans=KMeans()
visualize=KElbowVisualizer(kmeans,k=(1,16))
visualize.fit(features)
plt.suptitle("Elbow Graph and Elbow Visualizer")
visualize.poof()
plt.show()

# Calculate the silhouette score for different numbers of clusters
silhouette_avg=[]
for i in range(2,16):
  kmeans=KMeans(n_clusters=i)
  cluster_labels=kmeans.fit_predict(features)

  silhouette_avg.append(silhouette_score(features,cluster_labels))

# Plot the silhouette scores
plt.figure(figsize=(10,7))
plt.plot(range(2,16),silhouette_avg, 'bX-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis for optimal K')
plt.show()

# Perform KMeans clustering with the optimal number of clusters
model=KMeans(n_clusters=3)
model=model.fit(features)

# Predict the cluster labels
y_km=model.predict(features)
centers=model.cluster_centers_

# Add the cluster labels to the dataframe
df['Cluster']=pd.DataFrame(y_km)
df.to_csv("Cluster_data", index=False)

# Display the count of each cluster
df["Cluster"].value_counts()

# Plot a countplot of the cluster distribution
sns.countplot(data=df,x='Cluster')
plt.show()

# Load the clustered data from a CSV file
c_df=pd.read_csv("/content/Cluster_data")
c_df.head()

# Calculate the total search for each customer
c_df['Total Search']=c_df.iloc[:,3:38].sum(axis=1)

# Group the data by cluster and gender, and calculate the sum of total search
cl_0=c_df.groupby(['Cluster','Gender'],as_index=False).sum().query('Cluster==0')
cl_0

# Plot the customer count and total search for cluster 0
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=c_df.query('Cluster==0'), x='Gender')
plt.title( 'Customers count')

plt.subplot(1,2,2)
sns.barplot(data=cl_0,x='Gender',y='Total Search')
plt.title( 'Total Searches by Gender')
plt.suptitle( 'No. of customer and their total searches in "Cluster 0"')
plt.show()

# Repeat the above steps for cluster 1 and 2
cl_1=c_df.groupby(['Cluster','Gender'],as_index=False).sum().query('Cluster==1')
cl_1

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=c_df.query('Cluster==1'), x='Gender')
plt.title( 'Customers count')

plt.subplot(1,2,2)
sns.barplot(data=cl_1,x='Gender',y='Total Search')
plt.title( 'Total Searches by Gender')
plt.suptitle( 'No. of customer and their total searches in "Cluster 1"')
plt.show()

cl_2=c_df.groupby(['Cluster','Gender'],as_index=False).sum().query('Cluster==2')
cl_2

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
sns.countplot(data=c_df.query('Cluster==2'), x='Gender')
plt.title( 'Customers count')

plt.subplot(1,2,2)
sns.barplot(data=cl_2,x='Gender',y='Total Search')
plt.title( 'Total Searches by Gender')
plt.suptitle( 'No. of customer and their total searches in "Cluster 2"')
plt.show()

# Group the data by cluster and calculate the sum of total search and orders
final_df=c_df.groupby(['Cluster'],as_index=False).sum()
final_df

# Plot the total customers in each cluster
plt.figure(figsize=(15,6))
sns.countplot(data=c_df, x='Cluster',hue='Gender')
plt.title( 'Total Customers in each Cluster')
plt.show()

# Plot the total searches and past orders for each cluster
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.barplot(data=final_df,x='Cluster',y='Total Search')
plt.title( 'Total Searches by each group')

plt.subplot(1,2,2)
sns.barplot(data=final_df,x='Cluster',y='Orders')
plt.title( 'Past orders by each group')
plt.suptitle('No.of times customer searched the products and their past orders')
plt.show()