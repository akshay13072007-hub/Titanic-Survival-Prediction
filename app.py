import streamlit as st
from PIL import Image
import base64
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD IMAGE ----------------
def get_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

hero = get_base64("assets/hero.jpg")


# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- CSS ----------------
st.markdown(f"""
<style>

/* -----------------------------
   Hide Streamlit Default UI
------------------------------*/

#MainMenu {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

.stDeployButton {{
    display: none;
}}

/* -----------------------------
   Main App
------------------------------*/

.stApp {{
    background: linear-gradient(180deg,#04111d,#071b2d);
    color: white;
}}

.block-container {{
    max-width: 1300px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}}

/* -----------------------------
   Hero Banner
------------------------------*/

.hero {{
    width:100%;
    height:85vh;

    border-radius:28px;

    background:
        linear-gradient(
            rgba(0,0,0,.55),
            rgba(0,0,0,.75)
        ),
        url("data:image/jpg;base64,{hero}");

    background-size:cover;
    background-position:center;

    display:flex;
    justify-content:center;
    align-items:center;
    flex-direction:column;

    margin-bottom:35px;

    animation:fade 1s ease;
}}

.hero h1{{
    color:white;
    font-size:82px;
    font-weight:800;
    letter-spacing:2px;
    text-shadow:0 10px 30px rgba(0,0,0,.8);
    margin-bottom:10px;
}}

.hero p{{
    color:#d6d6d6;
    font-size:28px;
    letter-spacing:1px;
}}

@keyframes fade{{
    from{{
        opacity:0;
        transform:translateY(25px);
    }}

    to{{
        opacity:1;
        transform:translateY(0);
    }}
}}

/* -----------------------------
   Glass Cards
------------------------------*/

.glass{{
    background:rgba(255,255,255,.08);

    backdrop-filter:blur(20px);

    border:1px solid rgba(255,255,255,.15);

    border-radius:22px;

    padding:28px;

    box-shadow:0 10px 35px rgba(0,0,0,.35);

    transition:.35s;
}}

.glass:hover{{
    transform:translateY(-5px);
    box-shadow:0 15px 40px rgba(0,0,0,.45);
}}

/* -----------------------------
   Headings
------------------------------*/

h1,h2,h3{{
    color:white;
}}

label{{
    color:#e5e7eb !important;
    font-weight:600;
}}

/* -----------------------------
   Input Boxes
------------------------------*/

.stSelectbox > div > div,
.stNumberInput > div > div > input{{
    border-radius:12px !important;
}}

.stSlider > div > div > div > div{{
    background:#00aaff !important;
}}

/* -----------------------------
   PREMIUM BUTTON
------------------------------*/

div.stButton > button {{

    width:100%;

    height:60px;

    border:none;

    border-radius:15px;

    color:white;

    font-size:20px;

    font-weight:700;

    background:linear-gradient(
        135deg,
        #00A8FF,
        #005CFF
    );

    cursor:pointer;

    transition:all .3s ease;

    box-shadow:0 8px 25px rgba(0,140,255,.35);

}}

div.stButton > button:hover{{

    transform:translateY(-3px);

    background:linear-gradient(
        135deg,
        #00C6FF,
        #0072FF
    );

    box-shadow:0 15px 35px rgba(0,180,255,.55);

}}

div.stButton > button:active{{

    transform:scale(.98);

}}

div.stButton > button:focus{{

    outline:none !important;

    border:none !important;

}}

/* -----------------------------
   Success & Error Messages
------------------------------*/

.stSuccess{{
    border-radius:15px;
}}

.stError{{
    border-radius:15px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# 🚢 Titanic AI")

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.write("**Akshay S**")

    st.markdown("### 🤖 Model")
    st.write("Random Forest Classifier")

    st.markdown("### 🎯 Accuracy")
    st.write("**82.68%**")

    st.markdown("---")

    st.caption("Built with ❤️ using Python, Streamlit & Scikit-learn")

    # ---------------- HERO ----------------

st.markdown("""
<div class="hero">

<h1>🚢 Titanic Survival AI</h1>

<p>
Machine Learning Prediction System
</p>

</div>
""", unsafe_allow_html=True)

left,right = st.columns([1.2,1])

with left:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("## 📝 Passenger Details")

    pclass = st.selectbox(
        "🎟 Passenger Class",
        [1, 2, 3],
        help="1 = First Class | 2 = Second Class | 3 = Third Class"
    )

    sex = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "🎂 Age",
        1,
        80,
        25
    )

    fare = st.number_input(
        "💰 Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0,
        step=1.0
    )

    sibsp = st.number_input(
        "👨‍👩‍👧‍👦 Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "👨‍👩‍👧 Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    embarked = st.selectbox(
        "⚓ Port of Embarkation",
        ["S", "C", "Q"]
    )

    predict = st.button(
        "🚀 Predict Survival",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


with right:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("## 🤖 AI Prediction Assistant")

    if predict:

        import time

        with st.spinner("🤖 AI is analyzing passenger profile..."):
            time.sleep(1.2)

        sex_value = 0 if sex == "Male" else 1

        embarked_value = {
            "S": 0,
            "C": 1,
            "Q": 2
        }[embarked]

        input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_value],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_value]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        survive = probability[1] * 100
        not_survive = probability[0] * 100

        # -------------------------
        # Prediction Result
        # -------------------------

        if prediction == 1:

            st.success("### ✅ High Chance of Survival")

            risk = "🟢 LOW"

            explanation = f"""
The model predicts that this passenger has a **high probability of survival**.

### Why?

- Female / favorable passenger profile
- Age : **{age}**
- Passenger Class : **{pclass}**
- Ticket Fare : **₹ {fare:.2f}**

Passengers with similar characteristics
had better survival rates according to the
trained Random Forest model.
"""

            st.balloons()

        else:

            st.error("### ❌ Low Chance of Survival")

            risk = "🔴 HIGH"

            explanation = f"""
The model predicts that this passenger has a **low probability of survival**.

### Why?

- Male / higher risk passenger profile
- Age : **{age}**
- Passenger Class : **{pclass}**
- Ticket Fare : **₹ {fare:.2f}**

Passengers with similar characteristics
had lower survival rates according to the
trained Random Forest model.
"""

        st.markdown("---")

        st.markdown("### 🤖 AI Summary")

        st.markdown(explanation)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Survival Chance",
                f"{survive:.1f}%"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        st.markdown("")

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=survive,

            number={
                "suffix":"%"
            },

            title={
                "text":"Survival Probability"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"#00BFFF"
                },

                "steps":[

                    {
                        "range":[0,40],
                        "color":"#4d1f1f"
                    },

                    {
                        "range":[40,70],
                        "color":"#5c5316"
                    },

                    {
                        "range":[70,100],
                        "color":"#0f4d2c"
                    }

                ]

            }

        ))

        gauge.update_layout(

            height=330,

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="white"
            )

        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    else:

        st.info(
            """
### 🤖 AI Waiting...

Fill in the passenger details on the left and click **Predict Survival**.

The AI model will analyze the passenger profile and estimate the survival probability using a trained Random Forest Classifier.
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)