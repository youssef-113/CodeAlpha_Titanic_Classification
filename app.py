import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    model = joblib.load('Models/RandomForest_model.pkl')  # pre-trained pipeline
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('DataSet/datapreprocessed.csv')
    return df

model = load_model()
data = load_data()

# Extract features and labels
X = data.drop('Survived', axis=1)
y = data['Survived']

# Refit the model to ensure column alignment
model.fit(X, y)

st.title("🚢 Titanic Survival Predictor ")

# Sidebar for input features
st.sidebar.header("Passenger Details")

# Collect user input
def user_input_features():
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3], index=2)
    sex = st.sidebar.radio('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 0, 80, 30)
    sibsp = st.sidebar.slider('Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.sidebar.slider('Parents/Children Aboard', 0, 6, 0)
    fare = st.sidebar.slider('Fare', float(data['Fare'].min()), float(data['Fare'].max()), float(data['Fare'].mean()))
    embarked = st.sidebar.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

    # Additional engineered features (must match training pipeline)
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size
    is_lower_class = 1 if pclass == 3 else 0

    # Avoid qcut error by defining static bins like in training
    fare_bins = pd.qcut(data['Fare'], 4, labels=False, duplicates='drop')
    bin_edges = pd.qcut(data['Fare'], 4, retbins=True, duplicates='drop')[1]
    fare_bin = np.digitize(fare, bin_edges, right=True) - 1
    fare_bin = min(fare_bin, len(np.unique(fare_bins)) - 1)  # ensure index in valid range

    data_in = {
        'Pclass': pclass,
        'Sex': 0 if sex == 'female' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': {'S': 0, 'C': 1, 'Q': 2}[embarked],
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'FarePerPerson': fare_per_person,
        'Is_LowerClass': is_lower_class,
        'Fare_bin': fare_bin
    }
    features = pd.DataFrame(data_in, index=[0])
    return features

input_df = user_input_features()

# Align input_df with model's feature columns
missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[X.columns]  # Reorder columns to match training

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
prediction_proba = model.predict_proba(input_df)[0, 1]
prediction = model.predict(input_df)[0]

st.subheader('Prediction')
result = 'Survived ⚓' if prediction == 1 else 'Did Not Survive 💔'
st.write(f'The Passenger **{result}**')
st.write(f" with probability of **survival** is : **{prediction_proba:.2f}** %")
# --- Interactive Charts ---
st.subheader('Dataset Overview')

# Survival count chart
surv_chart = alt.Chart(data).mark_bar().encode(
    x=alt.X('Survived:N', title='Survival'),
    y=alt.Y('count()', title='Count'),
    color='Survived:N'
).properties(width=400, height=400)

# Age vs Fare scatter plot
age_hist = alt.Chart(data).mark_circle(size=60).encode(
    x='Age',
    y='Fare',
    color='Survived:N',
    tooltip=['Age', 'Fare', 'Survived']
).interactive().properties(width=400, height=300)


col1, col2 = st.columns(2, gap='large')
with col1:
    st.altair_chart(surv_chart, use_container_width=True)
with col2:
    st.altair_chart(age_hist, use_container_width=True)

# Footer
st.markdown('---')
st.write('Built with Youssef Bassiony ❤️ using Streamlit')
st.write('Check out my [GitHub](https://github.com/youssef-113) for more projects like this!')
