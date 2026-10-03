import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "SOCR-HeightWeight.csv"
CSS_FILE = BASE_DIR / "style.css"
HEIGHT_HTML = BASE_DIR / "height_card.html"
WEIGHT_HTML = BASE_DIR / "weight_card.html"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Height Weight Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD CSS
# =========================================================

with open(CSS_FILE, "r", encoding="utf-8") as file:
    css = file.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_FILE)

X = df[["Height(Inches)"]]
y = df["Weight(Pounds)"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# SCALER
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# MODELS
# =========================================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(alpha=0.1),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# TRAIN MODELS
# =========================================================

results = []
trained_models = {}


for name, model in models.items():

    if name in [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ]:

        model.fit(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    mse = mean_squared_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        prediction
    )

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])

    trained_models[name] = model


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]
)


# =========================================================
# BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    results_df["R2"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]


# =========================================================
# DATASET RANGE
# =========================================================

min_height = df["Height(Inches)"].min()
max_height = df["Height(Inches)"].max()


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "feet" not in st.session_state:
    st.session_state.feet = 5

if "inches" not in st.session_state:
    st.session_state.inches = 7


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="app-title">⚖️ Height → Weight Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">Machine Learning Based Weight Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="model-badge">🏆 Selected Model: {best_model_name}</div>',
    unsafe_allow_html=True
)


# =========================================================
# MAIN CARDS
# =========================================================

left_col, right_col = st.columns(
    2,
    gap="large"
)


# =========================================================
# HEIGHT CARD
# =========================================================

with left_col:

    with open(
        HEIGHT_HTML,
        "r",
        encoding="utf-8"
    ) as file:

        height_html = file.read()

    components.html(
        height_html,
        height=330,
        scrolling=False
    )

    st.write("")

    feet_col, inch_col = st.columns(2)

    with feet_col:

        feet = st.number_input(
            "Feet",
            min_value=3,
            max_value=8,
            value=st.session_state.feet,
            step=1
        )

    with inch_col:

        inches = st.number_input(
            "Inches",
            min_value=0,
            max_value=11,
            value=st.session_state.inches,
            step=1
        )

    st.markdown(
        f"""
<div class="height-value">
    {feet}' {inches}"
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# WEIGHT CARD
# =========================================================

with right_col:

    with open(
        WEIGHT_HTML,
        "r",
        encoding="utf-8"
    ) as file:

        weight_html = file.read()

    components.html(
        weight_html,
        height=330,
        scrolling=False
    )

    st.write("")

    if st.session_state.prediction is None:

        st.markdown(
            """
<div class="prediction-empty">
    -- kg
</div>

<div class="prediction-label">
    Enter height and click Predict
</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="prediction">
    {st.session_state.prediction:.2f} kg
</div>

<div class="prediction-label">
    Estimated Weight
</div>
""",
            unsafe_allow_html=True
        )


# =========================================================
# BUTTONS
# =========================================================

st.write("")

predict_col, clear_col, details_col = st.columns(3)


with predict_col:

    predict_button = st.button(
        "🚀 Predict Weight",
        use_container_width=True
    )


with clear_col:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


with details_col:

    details_button = st.button(
        "📊 Model Details",
        use_container_width=True
    )


# =========================================================
# PREDICT
# =========================================================

if predict_button:

    total_inches = (
        feet * 12
    ) + inches

    if (
        total_inches < min_height
        or
        total_inches > max_height
    ):

        st.warning(
            f"⚠️ Height is outside the dataset "
            f"range ({min_height:.2f} - "
            f"{max_height:.2f} inches)."
        )

    input_data = np.array(
        [[total_inches]]
    )

    if best_model_name in [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ]:

        input_data = scaler.transform(
            input_data
        )

    predicted_pounds = best_model.predict(
        input_data
    )[0]

    predicted_kg = (
        predicted_pounds * 0.45359237
    )

    st.session_state.prediction = predicted_kg
    st.session_state.feet = feet
    st.session_state.inches = inches

    st.rerun()


# =========================================================
# CLEAR
# =========================================================

if clear_button:

    st.session_state.prediction = None
    st.session_state.feet = 5
    st.session_state.inches = 7

    st.rerun()


# =========================================================
# MODEL DETAILS
# =========================================================

if details_button:

    st.write("")

    st.subheader(
        "📊 Regression Model Comparison"
    )

    display_df = results_df.copy()

    display_df["MAE"] = display_df["MAE"].round(3)
    display_df["MSE"] = display_df["MSE"].round(3)
    display_df["RMSE"] = display_df["RMSE"].round(3)
    display_df["R2"] = display_df["R2"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"Selected Model: {best_model_name}"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
    Height-Weight Regression • Machine Learning Project
</div>
""",
    unsafe_allow_html=True
)