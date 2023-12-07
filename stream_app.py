#load packages
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


st.title("Are you a LinkedIn user?")

def clean_sm(x):
        return np.where(x== 1, 1, 0)


s = pd.read_csv('social_media_usage.csv')
#import how we set the data 
ss= pd.DataFrame({
    'income': np.where((s["income"] >= 1) & (s["income"] <= 9), s["income"], np.nan),
    'education' :np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    'parent' : np.where(s['par'] == 'yes', 1, 0),
    'married': np.where(s['marital'] <= 3, 1, 0),
    'female': np.where(s['gender'] == 1, 1, 0),
    'age' : np.where(s['age'] <= 98, s['age'], np.nan),
    'sm_li': clean_sm(s['web1h'])
})

ss.dropna(subset=['income'], inplace=True)
ss.dropna(subset=['education'], inplace=True)
ss.dropna(subset=['age'], inplace=True)


   
#add the model here to train
y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')

# Fit algorithm to training data
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# add how the user inputs will be here


# user_inputs = pd.DataFrame({})
#age: use a slider for age 
age = st.slider('How old are you?', 0, 98, 25)
# st.write("I'm ", {age}, 'years old')

#education: drop down menu 
education = st.selectbox("Education level",
     ("Less than high school ",
     "High school incomplete ", 
     "High school graduate",
     "Some college",
     "Two-year associate degree",
     "Four-year college",
     "Some postgraduate",
     "8: Postgraduate"),
   index=None,
   placeholder="Select education level...",
)
#st.write(f"Education (pre-conversion): {education}")

if education == "Less than high school":
    education = 1
elif education == "High school incomplete":
    education = 2
elif education == "High school graduate":
    education = 3
elif education == "Some college":
    education = 4
elif education == "Two-year associate degree":
    education = 5
elif education == "Four-year college":
    education = 6
elif education == "Some postgraduate":
    education = 7
else:
    education = 8
    
#st.write(f"Education (post-conversion): {education}")


#income: drop down menu 
income = st.selectbox(
   "Household income?", 
   ("Less than $10,000", 
    "10 to under $20,000", 
    "20 to under $30,000",
    "30 to under $40,000",
    "40 to under $50,000",
    "50 to under $75,000",
    "75 to under $100,000",
    "8: 100 to under $150,000"),
   index=None,
   placeholder="Income level...",
)

#st.write(f"Income (pre-conversion): {income}")

if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
else:
    income= 8
#st.write(f"Income (post-conversion): {income}")


#female: drop down menu
female = st.selectbox(
   "Gender?",
   ("Female", "Male"),
   index=None,
   placeholder="Gender...",
)

#st.write('You selected:', female)

if female == female:
    female = 1
else:
    female = 0

#Maried: drop down menu 
married = st.selectbox(
   "Marital status",
   ("Married", "Single"),
   index=None,
   placeholder="Select contact method...",
)
if married == "Married":
    married = 1
else:
    married = 0

parent = st.selectbox(
   "Parent",
   ("Yes", "No"),
   index=None,
   placeholder="Select contact method...",
)

if parent == "Yes":
    parent = 1
else:
    parent = 0
#st.write('You selected:', married)

user_inputs = pd.DataFrame({
"income": [income],
"education": [education],
"parent": [parent],
"married": [married],
"female": [female],
"age": [age]
})


#Take all inputs and place into a dataframe to be sent through the model
#Model 
# are they a linkedin user
linkedin_user = lr.predict(user_inputs)
if linkedin_user == 1:
    linkedin_user = "a Linkedin User"
else:
    linkedin_user = "not a Linkedin User"
#the probability they are a user
Probability_user= lr.predict_proba(user_inputs)
rounded_probability = np.round(Probability_user, 2)
predicted_class = max(rounded_probability[0])
#Print the outputs
st.write(f"You are {linkedin_user}")
st.write(f"The probability of being a linkedin user is {predicted_class}")
