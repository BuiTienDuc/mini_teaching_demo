import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

st.markdown("""
<style>

/* Hide Streamlit default multipage navigation */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Optional: hide the divider line above it */
[data-testid="stSidebarNavSeparator"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Data Mining Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.big-icon {
    font-size: 56px;
    text-align: center;
    line-height: 1.1;
    margin-bottom: 6px;
}
.big-label {
    text-align: center;
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 12px;
}
.section-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.footer-text {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    padding-top: 8px;
    padding-bottom: 8px;
}
.small-note {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_from_excel(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name="sample_data", header=3)
    else:
        df = pd.read_excel("sample_data.xlsx", sheet_name="sample_data", header=3)

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    return df


with st.sidebar:
    st.markdown("# 📊 Data Mining Demo")

    st.markdown("### 👨‍🏫 Lecturer")
    st.write("**Bui Tien Duc**")
    st.write("📞 0769690731")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="big-icon">🖥️</div>', unsafe_allow_html=True)
        st.markdown('<div class="big-label">APP</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="big-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="big-label">DATA MINING</div>', unsafe_allow_html=True)

    st.page_link("app.py", label="Computerization", icon="🖥️")
    st.page_link("pages/data_mining.py", label="Data Mining", icon="📊")

    st.divider()

    st.markdown("### 🧭 Demo Flow")
    st.markdown("""
1. 📂 Data Collection  
2. 🧹 Preprocessing  
3. 🎯 Feature Selection  
4. 🧠 Model Selection  
5. ⚙️ Model Training  
6. 📏 Evaluation  
7. 📈 Visualization  
""")


st.title("📊 Data Mining Demo")
st.write("This demo uses the laptop e-commerce Excel dataset you provided.")

# Step 1
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("📂 Step 1: Data Collection")

uploaded_file = st.file_uploader("Upload Excel dataset", type=["xlsx"])

try:
    df = load_data_from_excel(uploaded_file)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Cannot read the Excel file correctly: {e}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.markdown('</div>', unsafe_allow_html=True)

# Step 2
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🧹 Step 2: Preprocessing")

st.write("The uploaded Excel file contains both numeric and categorical variables.")
st.write("Categorical columns will be encoded automatically for machine learning.")

numeric_cols_preview = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols_preview = df.select_dtypes(exclude=["number"]).columns.tolist()

c1, c2 = st.columns(2)
with c1:
    st.write("**Numeric columns:**")
    st.write(numeric_cols_preview)
with c2:
    st.write("**Categorical columns:**")
    st.write(categorical_cols_preview)

st.markdown('</div>', unsafe_allow_html=True)

# Step 3
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🎯 Step 3: Feature Selection")

problem_type = st.radio(
    "Choose problem type",
    ["Classification", "Regression"],
    horizontal=True
)

if problem_type == "Classification":
    default_target = "Purchased"
else:
    default_target = "Sales_Value_USD"

all_columns = df.columns.tolist()

if default_target not in all_columns:
    st.error(f"Target column '{default_target}' was not found in the dataset.")
    st.stop()

target = st.selectbox(
    "Select target",
    options=all_columns,
    index=all_columns.index(default_target)
)

default_features = [
    c for c in all_columns
    if c != target and c != "Customer_ID"
]

features = st.multiselect(
    "Select features",
    options=[c for c in all_columns if c != target],
    default=default_features
)

if len(features) == 0:
    st.error("Please select at least one feature.")
    st.stop()

st.write(f"**Selected target:** {target}")
st.write(f"**Selected features:** {features}")
st.markdown('</div>', unsafe_allow_html=True)

X = df[features].copy()
y = df[target].copy()

# Clean target
if problem_type == "Classification":
    y = pd.to_numeric(y, errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)
else:
    y = pd.to_numeric(y, errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(float)

if len(X) < 10:
    st.error("Not enough valid rows for training after preprocessing.")
    st.stop()

# Detect column types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessor
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Step 4
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🧠 Step 4: Model Selection")

if problem_type == "Classification":
    model_name = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )
else:
    model_name = st.selectbox(
        "Choose model",
        ["Linear Regression", "Random Forest Regressor"]
    )

test_size = st.slider("Test size", 0.2, 0.4, 0.3, 0.05)
st.markdown('</div>', unsafe_allow_html=True)

# Model
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
elif model_name == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Step 5
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("⚙️ Step 5: Training the Model")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Train/test split produced an empty dataset. Please check the data.")
        st.stop()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.success("Model training completed successfully.")
    st.write(f"Training rows: **{len(X_train)}** | Testing rows: **{len(X_test)}**")
except Exception as e:
    st.error(f"Training error: {e}")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# Step 6
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("📏 Step 6: Evaluation")

if problem_type == "Classification":
    y_pred_eval = pd.Series(y_pred).round().astype(int)

    acc = accuracy_score(y_test, y_pred_eval)
    prec = precision_score(y_test, y_pred_eval, zero_division=0)
    rec = recall_score(y_test, y_pred_eval, zero_division=0)
    f1 = f1_score(y_test, y_pred_eval, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_eval)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    st.pyplot(fig)

else:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("MAE", f"{mae:.2f}")

    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.subheader("Actual vs Predicted")
    st.dataframe(result_df.head(20), use_container_width=True)

    fig, ax = plt.subplots()
    ax.plot(result_df["Actual"].values, marker="o", label="Actual")
    ax.plot(result_df["Predicted"].values, marker="s", label="Predicted")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Sales Value")
    ax.legend()
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">🎓 Teaching Demo – Data Mining Application | Bui Tien Duc</div>',
    unsafe_allow_html=True
)