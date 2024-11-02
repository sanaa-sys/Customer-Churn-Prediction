from pandas.core.algorithms import value_counts
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
from openai import OpenAI
import utils as ut

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ['GROQ_API_KEY'])


def generate_email(probability, input_dict, explanation, surname, llm_model):
    prompt = f"""You are a manager at Headstarter Bank. You are responsible for ensuring customers have a good experience with your bank with offers and incentives. Here is the customer's information: {input_dict}. Here is the explanation  of the customer being risk at churning: {explanation}. Your task is to generate an email to the customer based on their information and the explanation. The explanation should be very brief. List out incentives and offers and services that the customer should be aware of in bullet points briefly. Each bullet point should be in a new line without italics. The email should end in the format:
"Ayesha Ali,
Manager, Headstarter Bank"  """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return response.choices[0].message.content


def explain_prediction(probability, input_dict, surname, llm_model):
    prompt = f"""You are an expert data scientist at Headstarter bank, where you specialize in interpreting and explaining ML models. 
    The ML model has predicted that a customer {surname} has a {round(probability * 100, 1)}% chance of churning, 
    based on the information below. Please explain the model's prediction in a way that is understandable to a bank customer.

    Here is the customer data:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:

    | Feature | Importance |
    |---|---|
    | CreditScore | 0.035005 |
    | Age | 0.109550 |
    | Tenure | 0.030054 |
    | Balance | 0.052786 |
    | NumOfProducts | 0.323888 |
    | HasCrCard | 0.031940 |
    | IsActiveMember | 0.164146 |
    | EstimatedSalary | 0.032655 |
    | Geography_France | 0.046463 |
    | Geography_Germany | 0.091373 |
    | Geography_Spain | 0.036855 |
    | Gender_Female | 0.045283 |
    | Gender_Male | 0.000000 |

    Here are summary characteristics for churned customers: {df[df['Exited'] == 1].describe()}

    Here are summary characteristics for non-churned customers: {df[df['Exited'] == 0].describe()}

    - If customer at over 40% risk of churning, explain in 3 sentences
    - If customer at under 40% risk of churning, explain in 3 sentence
    - Explain in detail based on customer information, summary statistics of churned customers and non-churned customers, feature importances and the machine learning model used without italics and bold. The font size should be consistent. 


    
    """
    print("Explanation Prompt", prompt)
    raw_response = client.chat.completions.create(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


def load_model(filename):
    with open(filename, 'rb') as file:
        return pk.load(file)


xgboostmodel = load_model("xgb_model.pkl")
nb_model = load_model("nb_model.pkl")
dt_model = load_model("dt_model.pkl")

svm_model = load_model("svm_model.pkl")
xgbboostmodelresampled = load_model("xgboost_model_resampled.pkl")
xgbboostmodelfeature = load_model("xgboost_modelFeature.pkl")
xgboostsmote = load_model("xgboost_modelSMOTE.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def predict_churn(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboostmodel.predict_proba(input_df)[0][1],
        'Naive Bayes': nb_model.predict_proba(input_df)[0][1],
        'Decision Tree': dt_model.predict_proba(input_df)[0][1],
    }
    avg_probabilities = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_guage_chart(avg_probabilities)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {round(avg_probabilities * 100, 1)}% chance of churning."
        )
    with col2:
        fig_prob = ut.create_bar_chart(probabilities)
        st.plotly_chart(fig_prob, use_container_width=True)

    return avg_probabilities


st.title("Customer Churn Prediction")
df = pd.read_csv("churn.csv")
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)
# initialize variables outside the if block
credit_score = None
location = None
gender = None
age = None
tenure = None
balance = None
num_of_products = None
has_credit_card = None
is_active_member = None
estimated_salary = None

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]
    # Remove col1, as all elements are in col2
    with st.columns(1)[0]:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=8000,
                                       value=int(
                                           selected_customer['CreditScore']))
        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                ["Spain", "France", "Germany"
                                 ].index(selected_customer['Geography']))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == "Male" else 1)
        age = st.number_input("Age",
                              min_value=18,
                              max_value=80,
                              value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure ",
                                 min_value=0,
                                 max_value=24,
                                 value=int(selected_customer['Tenure']))

        balance = st.number_input("Balance",
                                  min_value=0,
                                  max_value=1000000,
                                  value=int(selected_customer['Balance']))
        num_of_products = st.number_input(
            "Number of Products",
            min_value=0,
            max_value=10,
            value=int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member")

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

        available_llms = [
            "llama-3.2-3b-preview", "gemma2-9b-it", "mixtral-8x7b-32768",
            "llava-v1.5-7b-4096-preview"
        ]
        llm_model = st.selectbox("Select LLM Model", available_llms, index=0)

        # Now call the functions within the col2 block
        input_df, input_dict = prepare_input(credit_score, location, gender,
                                             age, tenure, balance,
                                             num_of_products, has_credit_card,
                                             is_active_member,
                                             estimated_salary)
        avg_probability = predict_churn(input_df, input_dict)
        explanation = explain_prediction(avg_probability, input_dict,
                                         selected_customer['Surname'],
                                         llm_model)
        st.markdown('---')
        st.subheader('Explanation of Prediction')
        st.markdown(explanation)
        email = generate_email(avg_probability, input_dict, explanation,
                               selected_customer['Surname'], llm_model)
        st.markdown('---')
        st.subheader('Your Email')
        st.markdown(email)
