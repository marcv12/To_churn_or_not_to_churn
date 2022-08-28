import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('churn.csv')

# website
# Transforming Categorical data to numerical data using LabelEncoder and OneHotEncoding

# Encoding categorical data that has 2 unique values
le = LabelEncoder()
le_count = 0
for col in dataset.columns[:]:
    if dataset[col].dtype == 'object' and col != "Attrition_Flag":
        if len(list(dataset[col].unique())) <= 2:
            le.fit(dataset[col])
            dataset[col] = le.transform(dataset[col])
            print('{} column was label encoded.'.format(col))


def to_numeric(s):
    if s == "Attrited Customer":
        return 1
    elif s == "Existing Customer":
        return 0


dataset["Attrition_Flag"] = dataset["Attrition_Flag"].apply(to_numeric)

# Encoding categorical data that has more than 2 unique values
data = dataset.drop(['CID', 'CLIENTNUM', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'],
                    axis=1)

Education_Level = pd.get_dummies(dataset.Education_Level).iloc[:, 1:]
Marital_Status = pd.get_dummies(dataset.Marital_Status).iloc[:, 1:]
Income_Category = pd.get_dummies(dataset.Income_Category).iloc[:, 1:]
Card_Category = pd.get_dummies(dataset.Card_Category).iloc[:, 1:]

data = pd.concat([data, Marital_Status, Income_Category, Card_Category, Education_Level], axis=1)

# Splitting data according to what we want to predict
X = data.drop(['Attrition_Flag'], axis=1)
y = data['Attrition_Flag']

# Splitting data from previous phase into train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rf_best_clf = RandomForestClassifier(bootstrap=False, max_depth=20, min_samples_split=2,
                                     n_estimators=800, max_features='sqrt', min_samples_leaf=1)
rf_best_clf.fit(X_train, y_train)
y_pred = rf_best_clf.predict(X_test)

# Save the model to disk
filename = 'model.sav'
joblib.dump(rf_best_clf, filename)
# load the model from disk
model = joblib.load(r"./model.sav")


def preprocess(df):
    le = LabelEncoder()
    for col in dataset.columns[:]:
        if dataset[col].dtype == 'object' and col != "Attrition_Flag":
            if len(list(dataset[col].unique())) <= 2:
                le.fit(dataset[col])
                dataset[col] = le.transform(dataset[col])

    def to_numeric(s):
        if s == "Attrited Customer":
            return 1
        elif s == "Existing Customer":
            return 0

    dataset["Attrition_Flag"] = dataset["Attrition_Flag"].apply(to_numeric)

    data = dataset.drop(['CID', 'CLIENTNUM', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'],
                        axis=1)

    Education_Level = pd.get_dummies(dataset.Education_Level).iloc[:, 1:]
    Marital_Status = pd.get_dummies(dataset.Marital_Status).iloc[:, 1:]
    Income_Category = pd.get_dummies(dataset.Income_Category).iloc[:, 1:]
    Card_Category = pd.get_dummies(dataset.Card_Category).iloc[:, 1:]

    data = pd.concat([data, Marital_Status, Income_Category, Card_Category, Education_Level], axis=1)

    X = data.drop(['Attrition_Flag'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def main():
    # Application title
    st.title('Attrited Customer App: Life made simple!')

    # Headline of app
    st.markdown("""
     Let's see how well our algorithm works! \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    st.sidebar.info('This app is created to predict Customer Churn')

    st.subheader("Dataset upload")
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Get first values of data
        st.write(data.head())
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        # Preprocess inputs
        preprocess_df = preprocess(data)
        if st.button('Predict'):
            # Get guess
            prediction = model.predict(preprocess_df)
            prediction_df = pd.DataFrame(prediction, columns=["Attrition_Flag"])
            prediction_df = prediction_df.replace({1: 'Yes, the customer will churn. Call them!',
                                                   0: 'No, the customer is happy with you. Good job!'})

            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.subheader('Prediction')
            st.write(prediction_df)


if __name__ == '__main__':
    main()