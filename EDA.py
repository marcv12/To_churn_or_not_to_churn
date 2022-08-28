import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import ListedColormap
pd.set_option('display.max_columns', None)

dataset = pd.read_csv('churn.csv')


# Data overview
def dataoveriew(df):
    print('Overview of dataset\n')
    print("What are the types of features?\n")
    print(dataset.dtypes)
    print("\nHow many missing values?", df.isnull().sum().values.sum())
    print("\nHow many unique values?")
    print(df.nunique())


dataoveriew(dataset)

dataset2 = dataset[['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon']]
dataset3 = dataset[['Credit_Limit', 'Avg_Utilization_Ratio']]

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns\n', horizontalalignment="center",
              fontsize=26)
for i in range(dataset2.shape[1]):
    plt.subplot(3, 2, i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#ec838a')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if vals >= 100:
        vals = 100
plt.show()

for i in range(dataset3.shape[1]):
    plt.subplot(1, 2, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    plt.hist(dataset3.iloc[:, i], bins=vals, color='#ec838a')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if vals >= 100:
        vals = 100
plt.show()

cat = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
plt.figure()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 25))
plt.suptitle('Bar Plots of Categorical Columns\n', horizontalalignment="center",
              fontsize=26)
for i, item in enumerate(cat):
    if i < 3:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i, 0], rot=0, color='#f3babc')

    elif i >= 3 and i < 6:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 3, 1], rot=0, color='#9b9c9a')

    elif i < 9:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 6, 2], rot=0, color='#ec838a')
plt.show()








# Explore target label
target_instance = dataset["Attrition_Flag"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Section'})
fig = px.pie(target_instance, values='Attrition_Flag', names='Section', title='Partition of Churn in dataset')
fig.show()

# Let's use bar chart for categorical features and histogram for numerical features
# Defining bar chart function
def bar(feature, df=dataset):
    # Groupby the categorical feature
    temp_df = df.groupby([feature, 'Attrition_Flag']).size().reset_index()
    temp_df = temp_df.rename(columns={0: 'Count'})
    # Calculate the value counts of each distribution and it's corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    # Calculate the value counts of each distribution and it's corresponding Percentages
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100, 1) for element in div_list]


    # Setting graph framework
    fig = px.bar(temp_df, x=feature, y='Count', color='Attrition_Flag', title=f'Churn rate by {feature}', barmode="group",
                 color_discrete_sequence=["green", "red"])

    return fig.show()

# Defining histogram function
def hist(feature):
    group_df = dataset.groupby([feature, 'Attrition_Flag']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    figure = px.histogram(group_df, x=feature, y='Count', color='Attrition_Flag', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    figure.show()

# Categorical demographic features
bar('Gender')
bar('Education_Level')
bar('Marital_Status')
bar('Income_Category')

# Numerical demographic features
hist('Customer_Age')
hist('Dependent_count')


# Categorical product features
bar('Card_Category')

# Defining the histogram plotting function


# Numerical product features
hist('Months_on_book')
hist('Total_Relationship_Count')
hist('Months_Inactive_12_mon')
hist('Contacts_Count_12_mon')
hist('Credit_Limit')
hist('Total_Trans_Amt')
hist('Total_Trans_Ct')
hist("Avg_Utilization_Ratio")

