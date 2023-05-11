#import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit


# Read the dataset
df = pd.read_csv('API_SP.DYN.IMRT.IN_DS2_en_csv_v2_5358355.csv', header=2)

# Functions From Practical class

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1960, 2010)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True) # replace NaN values with 0

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)

#finding the required years 
print(df_years.columns)



# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['1990'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('Infant mortality Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('Infant mortality rate(per 1,000 live births) in 1990')

# Add legend
plt.legend()

# Show the plot
plt.show()


# showing countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(cluster_countries)
    print()


def linear_model(x, a, b):
    return a*x + b


# Define the columns to use
columns_to_use = [str(year) for year in range(1960, 2009)]


# choose a country
country = 'Argentina'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values[0]
x_data = np.array(range(1960, 2009))
y_data = country_data

# Remove any NaN or inf values from y_data
y_data = np.nan_to_num(y_data)

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)


def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper


#showcasing Possible future values and corresponding confidence intervals
x_future = np.array(range(1960, 2041))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, color='green', alpha=0.3, label='Confidence range')
plt.xlabel('Year')
plt.ylabel('Infant mortality rate (per 1,000 live births)')
plt.title(f'{country} Infant Mortality Rate Fitting')
plt.legend()
plt.show()


# Filter the data to include only Iceland
iceland_data = df.loc[df['country'] == 'Iceland']

# Melt the dataframe to convert years into a single column
df_melt = pd.melt(iceland_data, id_vars=['country', 'country_code'], value_vars=[str(i) for i in range(2002, 2010)], var_name='year', value_name='infant_mortality_rate')

# Convert the value column to numeric
df_melt['infant_mortality_rate'] = pd.to_numeric(df_melt['infant_mortality_rate'], errors='coerce')

# Filter the data to include only the years in the range 2002-2009
df_melt = df_melt[(df_melt['year'] >= '2002') & (df_melt['year'] <= '2009')]

# Sort the data by year
df_melt = df_melt.sort_values(by='year')

# Plot the data as a histogram
plt.figure(figsize=(10, 6))
plt.hist(df_melt['infant_mortality_rate'], bins=10, edgecolor='black')
plt.title('Infant Mortality Rate in Iceland (2002-2009)')
plt.xlabel('Infant Mortality Rate')
plt.ylabel('Frequency')
plt.show()


