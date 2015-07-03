"""
    churn_predict.py

    Build a predictive model on taxi data to help determine whether or not a user will be retained.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

plt.style.use('ggplot')


# Perform any cleaning, exploratory analysis, and/or visualizations to use the provided data for this analysis.

raw_df = pd.read_csv([Insert .csv filename here])

## raw_df ##
# - city: city this user signed up in
# - phone: primary device for this user
# - signup_date: date of account registration; in the form `YYYYMMDD`
# - last_trip_date: the last time this user completed a trip; in the form `YYYYMMDD`
# - avg_dist: the average distance (in miles) per trip taken in the first 30 days after signup
# - avg_rating_by_driver: the rider’s average rating over all of their trips
# - avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips
# - surge_pct: the percent of trips taken with surge multiplier > 1*
# - avg_surge: The average surge multiplier over all of this user’s trips
# - trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
# - luxury_car_user: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise
# - weekday_pct: the percent of the user’s trips occurring during a weekday


print raw_df.head()
print
print raw_df.describe()

print
print raw_df.city.unique() #3 cities
print raw_df.phone.unique() #2 values + NaN's
print min(raw_df.signup_date) #2014-01-01
print max(raw_df.signup_date) #2014-01-31
print max(raw_df.last_trip_date) #2014-07-01
print min(raw_df.last_trip_date) #2014-01-01

axs = pd.scatter_matrix(raw_df, figsize=(12,12), alpha=.5, )
plt.show()

'''Question I have'''
#Users - are they weekday or weekend riders? (work vs. non-work/play)
#Luxary users - do they stay longer?
#Ride ourselves of single-serve users.. perhaps a "days since last ride"

'''Variables to explore'''
#Surge Rides = surge_pct * total rides
#Weekday Rides; Weekend Rides
#Signup Date Type - weekday vs. weekend?
#Luraxy Rides vs. Non_Luxary Rides

#High Surge usage - perhaps leads to customer dissatisfaction? churn?
#Distance? Perhaps we have some long drivers...
#City - are there demographic differences of the cities (by distance)?
#Date of last_trip for churn drivers - weekday or weekend?


low, high = min(raw_df['avg_dist']), max(raw_df['avg_dist'])

plt.figure(1, figsize=(8,5))
plt.hist(raw_df['avg_dist'], normed=True, bins=high-low+1)
plt.axis(xmax=20)
plt.xlabel("avg_dist")
plt.show()

plt.figure(2, figsize=(8,5))
plt.hist(raw_df['avg_dist'], normed=True, bins=high-low+1, cumulative=True)
plt.axis(xmax=20)
plt.xlabel("avg_dist")
plt.axhline(.95, c='r')
plt.show()


# Cleaning Data
print raw_df.info()
df = raw_df.copy(deep=True)

#Date Format
f = "%Y-%m-%d"

#convert strings to datetime
df['last_trip_date'] = pd.to_datetime(df['last_trip_date'], format=f)
df['signup_date'] = pd.to_datetime(df['signup_date'], format=f)

#Create the churn column
max_date = max(df['last_trip_date']) #2013-07-01
print (max_date - df['last_trip_date'][0]).days
df['churn'] = pd.Series([x.days > 30 for x in (max_date - df['last_trip_date'])]).astype(int)
print df['churn'].value_counts()

#Fill na's
df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean(), inplace=True)
df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean(), inplace=True)

#Convert Cities King's Landing" 'Astapor' 'Winterfell']
df['city_kl'] = (df['city'] == "King's Landing").astype(int)
df['city_a'] = (df['city'] == "Astapor").astype(int)
del df['city']

df['phone_iphone'] = (df['phone'] == 'iPhone').astype(int)
df['phone_android'] = (df['phone'] == 'Android').astype(int)
del df['phone']

#Feature Engineering
df['days_since_last_trip'] = [d.days for d in (max_date - df['last_trip_date'])]
df['one_and_done'] = (df['last_trip_date'] == df['signup_date']).astype(int)
df['luxury_car_user'] = df['luxury_car_user'].astype(int)

print sum(df['one_and_done'])
print
print df.columns

print df.head()


# K-Means algorithm test

df_tmp = df.copy(deep=True)
y = df_tmp.pop('churn').values
del df_tmp['avg_rating_by_driver']
del df_tmp['last_trip_date']
del df_tmp['signup_date']
del df_tmp['surge_pct']
del df_tmp['city_kl']
del df_tmp['city_a']
del df_tmp['phone_iphone']
del df_tmp['days_since_last_trip']

X = df_tmp.values
print X[0]
X_norm = scale(X)
print df_tmp.columns
print X_norm[0]

k_range = range(2,15)
list_of_inertia = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(X_norm)
    list_of_inertia.append(km.inertia_)

# Calculate information gain from initia
gain = [list_of_inertia[n-2]-list_of_inertia[n-1] for n in k_range[:-1]]
pct_gain = [list_of_inertia[n-2]/list_of_inertia[n-1]-1 for n in k_range[:-1]]

plt.plot(k_range, list_of_inertia)
plt.show()
plt.plot(k_range[1:], gain)
plt.show()
plt.plot(k_range[1:], pct_gain)
plt.show()

# Settled on 5 clusters
km = KMeans(n_clusters=5)
km.fit(X_norm)
list_of_inertia.append(km.inertia_)


for cluster in set(km.labels_):
    tmp_df = df[km.labels_==cluster]
    print "Cluster %i: %.3f percent of the %i people churn" % (cluster+1, tmp_df['churn'].sum()*1.0/tmp_df['churn'].count(), tmp_df['churn'].count())
print ""
for i, col in enumerate(df_tmp.columns):
    print col
    print "                          ", [round(km.cluster_centers_[k][i],3) for k in set(km.labels_)]
    print ""

'''Clusters:
    1 = hooked_early + weekday usage (power users)
    2 = iphone / non-luxury
    3 = luxery
    4 = android user
    5 = One & Doners
''''


# Final Model
clean_df = df.copy(deep=True)
clean_df['cluster_number'] = (km.labels_+ 1).astype(str)

y = clean_df.pop('churn').values
del clean_df['last_trip_date']
del clean_df['signup_date']
del clean_df['days_since_last_trip']
X = clean_df.values
X_train, X_test, y_train, y_test = train_test_split(X,y)

print clean_df.head()

# Run Gradient boosting classifier on train data
gbr = GradientBoostingClassifier()
gbr.fit(X_train,y_train)

key_model_features = sorted(zip(clean_df.columns, gbr.feature_importances_),key=lambda x: x[1], reverse=True)
for i in key_model_features:
    print i
print gbr.train_score_[-1]
plt.figure(1, figsize=(12,6))
plt.barh(range(len(gbr.feature_importances_)),gbr.feature_importances_, align='center')
plt.yticks(range(len(gbr.feature_importances_)), clean_df.columns)
plt.show()

# Using GBC model, show accuracy of model on test split
print gbr.score(X_test, y_test)

print float(df['churn'].sum())/df['churn'].count()

"We improved upon 'guessing everyone churned' by %.1f percentage points" % ((gbr.score(X_test, y_test) - float(df['churn'].sum())/df['churn'].count())*100)
