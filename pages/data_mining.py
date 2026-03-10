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

st.set_page_config(
    page_title="Data Mining Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
[data-testid="stSidebarNavSeparator"] {display:none;}

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
.metric-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
@st.cache_data
def load_data_from_excel(uploaded_file=None):
    """
    Expected structure based on the user's Excel:
    - sheet name: sample_data
    - real header row is the 4th row => header=3
    """
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name="sample_data", header=3)
    else:
        df = pd.read_excel("sample_data.xlsx", sheet_name="sample_data", header=3)

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_regression(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    return fig


def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(list(y_true), marker="o", label="Actual")
    ax.plot(list(y_pred), marker="s", label="Predicted")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Value")
    ax.legend()
    return fig


# =========================
# Sidebar
# =========================
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
4. 🧠 Model Comparison  
5. ⚙️ Model Training  
6. 📏 Evaluation  
7. 📈 Visualization  
8. 💡 Insight  
""")

# =========================
# Page Header
# =========================
st.title("📊 Data Mining Demo")
st.write("This version presents a more Data Mining workflow for the laptop e-commerce dataset.")

# =========================
# Step 1: Load data
# =========================
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

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 2: Preprocessing overview
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🧹 Step 2: Preprocessing Overview")

numeric_cols_preview = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols_preview = df.select_dtypes(exclude=["number"]).columns.tolist()

left, right = st.columns(2)
with left:
    st.subheader("Numeric Columns")
    st.write(numeric_cols_preview if numeric_cols_preview else "None")
with right:
    st.subheader("Categorical Columns")
    st.write(categorical_cols_preview if categorical_cols_preview else "None")

st.info("Missing numeric values will be filled with the median. Missing categorical values will be filled with the most frequent value. Categorical variables will be encoded automatically.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 3: Feature selection
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🎯 Step 3: Feature and Target Selection")

problem_type = st.radio(
    "Choose problem type",
    ["Classification", "Regression"],
    horizontal=True
)

default_target = "Purchased" if problem_type == "Classification" else "Sales_Value_USD"
all_columns = df.columns.tolist()

if default_target not in all_columns:
    st.error(f"Target column '{default_target}' was not found in the dataset.")
    st.stop()

target = st.selectbox(
    "Select target variable",
    options=all_columns,
    index=all_columns.index(default_target)
)

default_excluded = {"Customer_ID", target}
default_features = [c for c in all_columns if c not in default_excluded]

features = st.multiselect(
    "Select feature variables",
    options=[c for c in all_columns if c != target],
    default=default_features
)

if len(features) == 0:
    st.error("Please select at least one feature.")
    st.stop()

st.write(f"**Selected target:** {target}")
st.write(f"**Number of selected features:** {len(features)}")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prepare X, y
# =========================
X = df[features].copy()
y = df[target].copy()

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

if len(X) < 20:
    st.error("Not enough valid rows for training after preprocessing.")
    st.stop()

preprocessor, numeric_features, categorical_features = build_preprocessor(X)

# =========================
# Step 4: Model comparison
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🧠 Step 4: Model Comparison Setup")

if problem_type == "Classification":
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
else:
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42)
    }

test_size = st.slider("Test size", 0.2, 0.4, 0.3, 0.05)

show_comparison = st.checkbox("Compare all models automatically", value=True)

st.write("**Numeric features used:**", numeric_features if numeric_features else "None")
st.write("**Categorical features used:**", categorical_features if categorical_features else "None")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Train/Test split
# =========================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
except Exception as e:
    st.error(f"Train/test split error: {e}")
    st.stop()

if len(X_train) == 0 or len(X_test) == 0:
    st.error("The train/test split produced an empty dataset.")
    st.stop()

# =========================
# Step 5: Compare models
# =========================
comparison_df = None
best_model_name = None
best_pipeline = None
best_predictions = None

if show_comparison:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("⚙️ Step 5: Automatic Model Comparison")

    results = []

    for model_name, model in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if problem_type == "Classification":
            preds_eval = pd.Series(preds).round().astype(int)
            metrics = evaluate_classification(y_test, preds_eval)
            results.append({
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"]
            })
        else:
            metrics = evaluate_regression(y_test, preds)
            results.append({
                "Model": model_name,
                "R2": metrics["R2"],
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"]
            })

    comparison_df = pd.DataFrame(results)

    if problem_type == "Classification":
        comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
        best_model_name = comparison_df.iloc[0]["Model"]
    else:
        comparison_df = comparison_df.sort_values("R2", ascending=False).reset_index(drop=True)
        best_model_name = comparison_df.iloc[0]["Model"]

    best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", candidate_models[best_model_name])
    ])
    best_pipeline.fit(X_train, y_train)
    best_predictions = best_pipeline.predict(X_test)

    st.success(f"Best model selected automatically: **{best_model_name}**")
    st.dataframe(comparison_df, use_container_width=True)

    if problem_type == "Classification":
        chart_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].set_index("Model")
        st.bar_chart(chart_df)
    else:
        chart_df = comparison_df[["Model", "R2", "MSE", "MAE"]].set_index("Model")
        st.bar_chart(chart_df)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 6: Manual model choice
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🧠 Step 6: Final Model Selection")

default_index = list(candidate_models.keys()).index(best_model_name) if best_model_name else 0
selected_model_name = st.selectbox(
    "Choose the final model for presentation",
    options=list(candidate_models.keys()),
    index=default_index
)

final_model = candidate_models[selected_model_name]
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", final_model)
])

final_pipeline.fit(X_train, y_train)
final_pred = final_pipeline.predict(X_test)

st.success(f"Final model trained successfully: **{selected_model_name}**")
st.write(f"Training rows: **{len(X_train)}** | Testing rows: **{len(X_test)}**")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 7: Evaluation
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("📏 Step 7: Evaluation")

if problem_type == "Classification":
    final_pred_eval = pd.Series(final_pred).round().astype(int)
    metrics = evaluate_classification(y_test, final_pred_eval)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    c2.metric("Precision", f"{metrics['Precision']:.2f}")
    c3.metric("Recall", f"{metrics['Recall']:.2f}")
    c4.metric("F1-Score", f"{metrics['F1-Score']:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, final_pred_eval)
    st.pyplot(plot_confusion_matrix(cm))

else:
    metrics = evaluate_regression(y_test, final_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics['R2']:.2f}")
    c2.metric("MSE", f"{metrics['MSE']:.2f}")
    c3.metric("MAE", f"{metrics['MAE']:.2f}")

    st.subheader("Actual vs Predicted")
    result_df = pd.DataFrame({
        "Actual": list(y_test),
        "Predicted": list(final_pred)
    })
    st.dataframe(result_df.head(20), use_container_width=True)
    st.pyplot(plot_actual_vs_predicted(y_test, final_pred))

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 8: Feature importance
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("📈 Step 8: Feature Importance / Insight")

final_estimator = final_pipeline.named_steps["model"]

if hasattr(final_estimator, "feature_importances_"):
    try:
        transformed_feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = final_estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": transformed_feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        st.subheader("Top Important Features")
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"Feature importance could not be displayed: {e}")
else:
    st.info("Feature importance is not directly available for this model. Try a tree-based model such as Decision Tree or Random Forest.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Step 9: Teaching summary
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("💡 Teaching Summary")

st.write("""
This data mining demo shows a complete Data Mining workflow:

- reading a structured Excel dataset
- preprocessing numeric and categorical variables
- selecting features and target
- comparing multiple models
- training a final model
- evaluating performance
- visualizing results and feature importance

This makes the demo stronger for teaching interviews because it connects theory, Data Mining workflow, and business interpretation.
""")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">🎓 Teaching Demo – Data Mining and Its Application | Bui Tien Duc</div>',
    unsafe_allow_html=True
)