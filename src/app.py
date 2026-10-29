import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.base import clone
import requests

st.set_page_config(
    page_title="InsightX360",
    page_icon="📊",
    layout="wide",
)

st.title("📊 InsightX360")
st.caption(
    "Intelligent Business Analytics, Predictive Modeling "
    "& Quality Assurance Platform"
)

with st.sidebar:
    st.title("InsightX360")
    page = st.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Business Analytics & Data Quality",
            "Customer Churn & Retention",
            "API Testing & Defect Intelligence",
        ],
    )
    st.divider()
    st.caption("Development Version 1.1")


# --------------------------------------------------
# Home Dashboard
# --------------------------------------------------
if page == "Home Dashboard":
    st.header("Home Dashboard")
    st.info("The KPI values below are illustrative demo data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", "₹12.8 L", "+8.4%")
    c2.metric("Customers", "2,450", "+5.2%")
    c3.metric("Churn Rate", "12.4%", "-1.8%")
    c4.metric("API Test Pass Rate", "96.2%", "+2.1%")

    demo = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Revenue (₹ lakh)": [8.2, 9.1, 8.8, 10.4, 11.6, 12.8],
        "Expenses (₹ lakh)": [5.4, 5.8, 5.6, 6.2, 6.8, 7.1],
    })

    st.subheader("Business Performance")
    fig = px.line(
        demo,
        x="Month",
        y=["Revenue (₹ lakh)", "Expenses (₹ lakh)"],
        markers=True,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Sample data only; not actual business results.")


# --------------------------------------------------
# Business Analytics & Data Quality
# --------------------------------------------------
elif page == "Business Analytics & Data Quality":
    st.header("📈 Business Analytics & Data Quality")
    st.write(
        "Upload a CSV file, configure validation checks, "
        "and download a detailed data-quality report."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin your analysis.")
    else:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty or len(df.columns) == 0:
                st.warning("The uploaded CSV contains no data rows or columns.")
                st.stop()

            total_rows, total_columns = df.shape
            total_cells = total_rows * total_columns

            st.success("Dataset loaded successfully!")

            st.subheader("1. Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{total_rows:,}")
            c2.metric("Columns", f"{total_columns:,}")
            c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
            c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

            with st.expander("Preview dataset", expanded=True):
                st.dataframe(df.head(100), width="stretch")

            # ------------------------------------------
            # Configure validation rules
            # ------------------------------------------
            st.subheader("2. Configure Validation Checks")

            check_missing = st.checkbox(
                "Check for missing values",
                value=True,
            )
            check_duplicates = st.checkbox(
                "Check for duplicate rows",
                value=True,
            )
            check_ranges = st.checkbox(
                "Check numeric values against an allowed range",
                value=False,
            )

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            selected_column = None
            lower_limit = None
            upper_limit = None

            if check_ranges:
                if not numeric_columns:
                    st.warning(
                        "No numeric columns are available for range checking."
                    )
                    check_ranges = False
                else:
                    selected_column = st.selectbox(
                        "Select a numeric column",
                        numeric_columns,
                    )

                    valid_values = df[selected_column].dropna()

                    if valid_values.empty:
                        st.warning(
                            "The selected column has no numeric values to check."
                        )
                        check_ranges = False
                    else:
                        observed_min = float(valid_values.min())
                        observed_max = float(valid_values.max())

                        left, right = st.columns(2)

                        with left:
                            lower_limit = st.number_input(
                                "Allowed minimum",
                                value=observed_min,
                                format="%.4f",
                            )

                        with right:
                            upper_limit = st.number_input(
                                "Allowed maximum",
                                value=observed_max,
                                format="%.4f",
                            )

                        if lower_limit > upper_limit:
                            st.error(
                                "The allowed minimum cannot exceed the maximum."
                            )
                            st.stop()

            # ------------------------------------------
            # Run validation checks
            # ------------------------------------------
            st.subheader("3. Data Quality Results")

            missing_cells = int(df.isna().sum().sum())
            duplicate_rows = int(df.duplicated().sum())

            missing_by_column = df.isna().sum()
            duplicate_mask = df.duplicated()

            range_violations = 0
            range_violation_rows = pd.Series(
                False, index=df.index
            )

            if check_ranges and selected_column is not None:
                values = df[selected_column]
                range_violation_rows = (
                    values.notna()
                    & (
                        (values < lower_limit)
                        | (values > upper_limit)
                    )
                )
                range_violations = int(range_violation_rows.sum())

            # Count issues only from enabled checks.
            issue_count = 0

            if check_missing:
                issue_count += missing_cells

            if check_duplicates:
                issue_count += duplicate_rows

            if check_ranges:
                issue_count += range_violations

            quality_score = max(
                0.0,
                100.0 * (1 - issue_count / total_cells),
            )

            quality_score = round(quality_score, 2)

            if quality_score >= 90:
                quality_status = "Good"
            elif quality_score >= 70:
                quality_status = "Needs Improvement"
            else:
                quality_status = "Poor"

            score_col, status_col, issue_col = st.columns(3)

            score_col.metric(
                "Data Quality Score",
                f"{quality_score:.2f}/100",
            )
            status_col.metric("Quality Status", quality_status)
            issue_col.metric("Issues Found", f"{issue_count:,}")

            st.progress(quality_score / 100)

            st.caption(
                "Custom score = max(0, 100 × (1 − enabled issues / total cells)). "
                "Each missing cell, duplicate row, and out-of-range cell "
                "counts as one issue when its check is enabled. "
                "This is a project-specific indicator, not a universal standard."
            )

            # ------------------------------------------
            # Detailed issue results
            # ------------------------------------------
            report_rows = []

            if check_missing:
                missing_columns = int((missing_by_column > 0).sum())

                report_rows.append({
                    "Validation Check": "Missing values",
                    "Status": "PASS" if missing_cells == 0 else "ISSUE",
                    "Issue Count": missing_cells,
                    "Severity": (
                        "None" if missing_cells == 0
                        else "High" if missing_cells / total_cells >= 0.10
                        else "Medium"
                    ),
                    "Details": (
                        "No missing cells found."
                        if missing_cells == 0
                        else f"{missing_columns} column(s) contain missing values."
                    ),
                })

            if check_duplicates:
                report_rows.append({
                    "Validation Check": "Duplicate rows",
                    "Status": "PASS" if duplicate_rows == 0 else "ISSUE",
                    "Issue Count": duplicate_rows,
                    "Severity": (
                        "None" if duplicate_rows == 0
                        else "High" if duplicate_rows / total_rows >= 0.10
                        else "Medium"
                    ),
                    "Details": (
                        "No duplicate rows found."
                        if duplicate_rows == 0
                        else "Rows repeated after their first occurrence."
                    ),
                })

            if check_ranges and selected_column is not None:
                report_rows.append({
                    "Validation Check": "Numeric range",
                    "Status": "PASS" if range_violations == 0 else "ISSUE",
                    "Issue Count": range_violations,
                    "Severity": (
                        "None" if range_violations == 0
                        else "High" if range_violations / total_rows >= 0.10
                        else "Medium"
                    ),
                    "Details": (
                        f"{selected_column}: allowed range "
                        f"{lower_limit} to {upper_limit}."
                    ),
                })

            quality_report = pd.DataFrame(report_rows)

            if quality_report.empty:
                st.info("Enable at least one validation check to see results.")
            else:
                st.dataframe(
                    quality_report,
                    width="stretch",
                    hide_index=True,
                )

            # ------------------------------------------
            # Missing values chart
            # ------------------------------------------
            if check_missing:
                st.subheader("4. Missing Values by Column")

                missing_report = pd.DataFrame({
                    "Column": df.columns.astype(str),
                    "Missing Count": df.isna().sum().values,
                })
                missing_report["Missing %"] = (
                    missing_report["Missing Count"] / total_rows * 100
                ).round(2)

                missing_report = missing_report[
                    missing_report["Missing Count"] > 0
                ]

                if missing_report.empty:
                    st.success("No missing values detected.")
                else:
                    st.dataframe(
                        missing_report,
                        width="stretch",
                        hide_index=True,
                    )
                    fig = px.bar(
                        missing_report,
                        x="Column",
                        y="Missing Count",
                        title="Missing Values by Column",
                    )
                    st.plotly_chart(fig, width="stretch")

            # ------------------------------------------
            # Numeric summary
            # ------------------------------------------
            st.subheader("5. Numerical Summary")

            numeric_df = df.select_dtypes(include="number")

            if not numeric_df.empty:
                st.dataframe(
                    numeric_df.describe().round(2),
                    width="stretch",
                )
            else:
                st.info("No numeric columns were found.")

            # ------------------------------------------
            # Download detailed report
            # ------------------------------------------
            st.subheader("6. Download Reports")

            column_report = pd.DataFrame({
                "Column": df.columns.astype(str),
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
                "Missing %": (
                    df.isna().mean().values * 100
                ).round(2),
                "Unique Values": df.nunique(dropna=True).values,
            })

            csv_report = column_report.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download column quality report",
                data=csv_report,
                file_name="insightx360_column_quality_report.csv",
                mime="text/csv",
            )

            detailed_report = quality_report.copy()
            detailed_report["Data Quality Score"] = quality_score
            detailed_report["Quality Status"] = quality_status

            st.download_button(
                "Download validation results",
                data=detailed_report.to_csv(index=False).encode("utf-8"),
                file_name="insightx360_validation_results.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(
                "The CSV could not be processed. Please check its format "
                "and try again."
            )
            st.caption(f"Technical details: {error}")


# --------------------------------------------------
# Customer Churn & Retention
# --------------------------------------------------
elif page == "Customer Churn & Retention":
    st.header("🎯 Customer Churn & Retention Analytics")
    st.write(
        "Analyze customer behavior, understand churn patterns, "
        "and identify retention opportunities."
    )

    # Load the default project dataset or an uploaded CSV.
    default_file = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "customer_churn.csv"
    )

    uploaded_file = st.file_uploader(
        "Upload a customer churn CSV (optional)",
        type=["csv"],
        key="churn_csv_upload",
    )

    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.caption("Source: uploaded customer dataset")
        elif default_file.exists():
            df = pd.read_csv(default_file)
            st.caption("Source: data/customer_churn.csv")
        else:
            st.error(
                "Customer dataset not found. Please create "
                "data/customer_churn.csv or upload a CSV file."
            )
            st.stop()

        required_columns = [
            "CustomerID",
            "TenureMonths",
            "MonthlyCharges",
            "Contract",
            "SupportCalls",
            "Churn",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error(
                "The dataset is missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if df.empty:
            st.warning("The customer dataset contains no records.")
            st.stop()

        # Clean the churn labels.
        df["Churn"] = (
            df["Churn"]
            .astype("string")
            .str.strip()
            .str.title()
        )

        valid_labels = {"Yes", "No"}
        invalid_labels = (
            df["Churn"].notna()
            & ~df["Churn"].isin(valid_labels)
        )

        if df["Churn"].isna().any() or invalid_labels.any():
            st.error(
                "The Churn column must contain only Yes or No, "
                "with no missing values."
            )
            st.stop()

        total_customers = len(df)
        churned_customers = int((df["Churn"] == "Yes").sum())
        retained_customers = int((df["Churn"] == "No").sum())
        churn_rate = churned_customers / total_customers * 100
        retention_rate = retained_customers / total_customers * 100

        st.subheader("1. Customer Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Customers Churned", f"{churned_customers:,}")
        c3.metric("Churn Rate", f"{churn_rate:.2f}%")
        c4.metric("Retention Rate", f"{retention_rate:.2f}%")

        st.progress(churn_rate / 100)
        st.caption(
            "Churn rate = customers marked Yes ÷ total customers × 100. "
            "These results describe the supplied dataset, not a forecast."
        )

        st.subheader("2. Customer Data Preview")
        st.dataframe(df.head(100), width="stretch")

        st.subheader("3. Churn Distribution")

        churn_summary = pd.DataFrame({
            "Customer Status": ["Churned", "Retained"],
            "Customers": [churned_customers, retained_customers],
        })

        fig = px.pie(
            churn_summary,
            names="Customer Status",
            values="Customers",
            title="Churned vs Retained Customers",
            hole=0.4,
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("4. Churn by Contract Type")

        contract_summary = (
            df.groupby("Contract", dropna=False)
            .agg(
                Total_Customers=("CustomerID", "size"),
                Churned_Customers=(
                    "Churn",
                    lambda values: (values == "Yes").sum(),
                ),
            )
            .reset_index()
        )

        contract_summary["Churn Rate (%)"] = (
            contract_summary["Churned_Customers"]
            / contract_summary["Total_Customers"]
            * 100
        ).round(2)

        st.dataframe(
            contract_summary,
            width="stretch",
            hide_index=True,
        )

        fig = px.bar(
            contract_summary,
            x="Contract",
            y="Churn Rate (%)",
            color="Contract",
            title="Churn Rate by Contract Type",
            text="Churn Rate (%)",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("5. Customer Behavior")

        left, right = st.columns(2)

        with left:
            fig = px.box(
                df,
                x="Churn",
                y="TenureMonths",
                color="Churn",
                title="Customer Tenure vs Churn",
                labels={"TenureMonths": "Tenure (months)"},
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            fig = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges vs Churn",
                labels={"MonthlyCharges": "Monthly Charges"},
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("6. Initial Retention Insights")

        month_to_month = contract_summary[
            contract_summary["Contract"] == "Month-to-month"
        ]

        if not month_to_month.empty:
            monthly_rate = float(
                month_to_month["Churn Rate (%)"].iloc[0]
            )
            st.write(
                f"- Month-to-month contract churn rate: "
                f"**{monthly_rate:.2f}%**."
            )

        churned = df[df["Churn"] == "Yes"]

        if not churned.empty:
            avg_tenure = churned["TenureMonths"].mean()
            avg_charges = churned["MonthlyCharges"].mean()

            st.write(
                f"- Average tenure among churned customers: "
                f"**{avg_tenure:.1f} months**."
            )
            st.write(
                f"- Average monthly charges among churned customers: "
                f"**{avg_charges:.2f}**."
            )

        st.info(
            "These are descriptive insights from a small sample. "
            "They do not establish the causes of churn. Predictive "
            "modeling and more advanced retention recommendations "
            "will be added in the next stage."
        )

        # ------------------------------------------
        # 7. Machine Learning: Churn Prediction
        # ------------------------------------------
        st.subheader("7. Customer Churn Prediction Model")

        st.write(
            "Train a Logistic Regression model using historical "
            "customer data, evaluate it on a held-out test set, "
            "and estimate churn risk for a new customer."
        )

        numeric_features = [
            "TenureMonths",
            "MonthlyCharges",
            "SupportCalls",
        ]
        categorical_features = ["Contract"]
        feature_columns = numeric_features + categorical_features

        # Convert input columns to suitable data types.
        model_df = df[
            feature_columns + ["Churn"]
        ].copy()

        for column in numeric_features:
            model_df[column] = pd.to_numeric(
                model_df[column], errors="coerce"
            )

        model_df["Contract"] = (
            model_df["Contract"]
            .astype("string")
            .str.strip()
        )

        model_df = model_df.dropna(
            subset=feature_columns + ["Churn"]
        )

        if model_df["Churn"].nunique() < 2:
            st.warning(
                "The model needs historical examples of both "
                "churned and retained customers."
            )
        elif len(model_df) < 12 or model_df["Churn"].value_counts().min() < 2:
            st.warning(
                "There are too few usable records in each churn class "
                "for this train/test demonstration. Add more historical "
                "customer records before training."
            )
        else:
            X = model_df[feature_columns]
            y = model_df["Churn"]

            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore"
            )

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        numeric_transformer,
                        numeric_features,
                    ),
                    (
                        "categorical",
                        categorical_transformer,
                        categorical_features,
                    ),
                ]
            )

            evaluation_model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )

            evaluation_model.fit(X_train, y_train)
            y_pred = evaluation_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(
                y_test, y_pred, pos_label="Yes", zero_division=0
            ) * 100
            recall = recall_score(
                y_test, y_pred, pos_label="Yes", zero_division=0
            ) * 100

            st.markdown("**Model evaluation on unseen test records**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy:.1f}%")
            m2.metric("Churn Precision", f"{precision:.1f}%")
            m3.metric("Churn Recall", f"{recall:.1f}%")

            st.caption(
                f"Training records: {len(X_train)} | "
                f"Test records: {len(X_test)}. "
                "Scores may vary substantially with such a small dataset."
            )

            # Fit a final version on all available historical data
            # for the interactive customer prediction below.
            prediction_model = clone(evaluation_model)
            prediction_model.fit(X, y)

            st.markdown("**Estimate churn risk for a customer**")

            with st.form("churn_prediction_form"):
                p1, p2 = st.columns(2)

                with p1:
                    customer_tenure = st.number_input(
                        "Tenure (months)",
                        min_value=0,
                        max_value=120,
                        value=6,
                        step=1,
                    )
                    customer_charges = st.number_input(
                        "Monthly charges",
                        min_value=0.0,
                        max_value=10000.0,
                        value=85.0,
                        step=5.0,
                    )

                with p2:
                    customer_contract = st.selectbox(
                        "Contract type",
                        ["Month-to-month", "One-year", "Two-year"],
                    )
                    customer_support_calls = st.number_input(
                        "Support calls",
                        min_value=0,
                        max_value=100,
                        value=3,
                        step=1,
                    )

                predict_clicked = st.form_submit_button(
                    "Estimate Churn Risk"
                )

            if predict_clicked:
                new_customer = pd.DataFrame([{
                    "TenureMonths": customer_tenure,
                    "MonthlyCharges": customer_charges,
                    "SupportCalls": customer_support_calls,
                    "Contract": customer_contract,
                }])

                churn_probability = prediction_model.predict_proba(
                    new_customer[feature_columns]
                )[
                    0,
                    list(
                        prediction_model.named_steps[
                            "classifier"
                        ].classes_
                    ).index("Yes"),
                ] * 100

                prediction = (
                    "Higher estimated risk"
                    if churn_probability >= 60
                    else "Moderate estimated risk"
                    if churn_probability >= 30
                    else "Lower estimated risk"
                )

                st.metric(
                    "Estimated churn probability",
                    f"{churn_probability:.1f}%",
                )
                st.write(f"**Risk category:** {prediction}")

                st.caption(
                    "This probability is a model estimate, not a "
                    "guarantee or a calibrated real-world probability. "
                    "The model uses a tiny synthetic dataset and should "
                    "not be used to make actual customer decisions."
                )


            # ------------------------------------------
            # 8. Customer Risk Report and Retention Actions
            # ------------------------------------------
            st.subheader("8. Customer Risk & Retention Report")

            risk_features = model_df[feature_columns]
            churn_class_index = list(
                prediction_model.named_steps[
                    "classifier"
                ].classes_
            ).index("Yes")

            risk_probabilities = (
                prediction_model.predict_proba(risk_features)[
                    :, churn_class_index
                ] * 100
            )

            risk_report = pd.DataFrame({
                "CustomerID": df.loc[
                    model_df.index, "CustomerID"
                ].astype(str).values,
                "TenureMonths": model_df[
                    "TenureMonths"
                ].values,
                "Contract": model_df["Contract"].values,
                "MonthlyCharges": model_df[
                    "MonthlyCharges"
                ].values,
                "Actual Churn": model_df["Churn"].values,
                "Estimated Churn Risk (%)": risk_probabilities.round(2),
            })

            risk_report["Risk Category"] = risk_report[
                "Estimated Churn Risk (%)"
            ].apply(
                lambda risk: (
                    "Higher"
                    if risk >= 60
                    else "Moderate"
                    if risk >= 30
                    else "Lower"
                )
            )

            risk_report["Suggested Action"] = risk_report[
                "Risk Category"
            ].map({
                "Higher": (
                    "Prioritize a support check-in and review "
                    "contract or service concerns."
                ),
                "Moderate": (
                    "Check customer satisfaction and offer "
                    "relevant support or plan guidance."
                ),
                "Lower": (
                    "Maintain service quality and routine engagement."
                ),
            })

            high_risk_count = int(
                (risk_report["Risk Category"] == "Higher").sum()
            )
            moderate_risk_count = int(
                (risk_report["Risk Category"] == "Moderate").sum()
            )
            lower_risk_count = int(
                (risk_report["Risk Category"] == "Lower").sum()
            )

            r1, r2, r3 = st.columns(3)
            r1.metric("Higher Risk", high_risk_count)
            r2.metric("Moderate Risk", moderate_risk_count)
            r3.metric("Lower Risk", lower_risk_count)

            st.dataframe(
                risk_report.sort_values(
                    "Estimated Churn Risk (%)",
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download customer risk and retention report",
                data=risk_report.to_csv(index=False).encode("utf-8"),
                file_name="insightx360_customer_risk_report.csv",
                mime="text/csv",
            )

            st.caption(
                "Risk categories use project-defined thresholds: "
                "Higher ≥ 60%, Moderate ≥ 30%, Lower < 30%. "
                "These scores are generated on the same small sample "
                "used to fit the final model, so they are exploratory "
                "and may be overoptimistic. Validate on larger, "
                "representative historical data before business use."
            )

        st.download_button(
            "Download churn analysis",
            data=contract_summary.to_csv(index=False).encode("utf-8"),
            file_name="insightx360_churn_analysis.csv",
            mime="text/csv",
        )

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty.")
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read the customer dataset: {error}")




# --------------------------------------------------
# API Testing & Defect Intelligence
# --------------------------------------------------
elif page == "API Testing & Defect Intelligence":

    st.header("🧪 API Testing & Defect Intelligence")

    st.write(
        "Send HTTP requests, inspect responses, validate API "
        "behavior, and automatically generate defect reports."
    )

    st.info(
        "Start with the public JSONPlaceholder test API. "
        "Use only APIs you own or are authorized to test."
    )

    # --------------------------------------------------
    # API Test Form
    # --------------------------------------------------

    with st.form("api_test_form"):

        method = st.selectbox(
            "HTTP Method",
            ["GET", "POST"],
        )

        url = st.text_input(
            "API Endpoint URL",
            value="https://jsonplaceholder.typicode.com/posts/1",
        )

        request_body = st.text_area(
            "Request Body (JSON, used for POST)",
            value=(
                '{\n'
                '  "title": "InsightX360 test",\n'
                '  "body": "Testing an API request",\n'
                '  "userId": 1\n'
                '}'
            ),
            height=140,
        )

        run_test = st.form_submit_button(
            "Run API Test"
        )

    # --------------------------------------------------
    # Execute API Test
    # --------------------------------------------------

    if run_test:

        from urllib.parse import urlparse
        import json
        import time

        parsed_url = urlparse(url.strip())

        # Validate URL
        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.netloc
        ):

            st.error(
                "Enter a valid URL beginning with "
                "http:// or https://."
            )

        else:

            payload = None
            body_is_valid = True

            # Validate POST JSON
            if method == "POST":

                try:

                    payload = json.loads(request_body)

                    if not isinstance(payload, dict):

                        st.error(
                            "The POST request body must be "
                            "a JSON object."
                        )

                        body_is_valid = False

                except json.JSONDecodeError as error:

                    st.error(
                        f"Invalid JSON request body: {error}"
                    )

                    body_is_valid = False

            # Continue only if request input is valid
            if body_is_valid:

                try:

                    # ----------------------------------
                    # Send API Request
                    # ----------------------------------

                    start_time = time.perf_counter()

                    response = requests.request(
                        method=method,
                        url=url.strip(),
                        json=payload if method == "POST" else None,
                        timeout=10,
                    )

                    elapsed_ms = (
                        time.perf_counter() - start_time
                    ) * 1000

                    # ----------------------------------
                    # Prepare Response Data
                    # ----------------------------------

                    try:

                        response_data = response.json()

                    except ValueError:

                        response_data = None

                    content_type = response.headers.get(
                        "Content-Type",
                        "",
                    )

                    # ----------------------------------
                    # API Assertion Checks
                    # ----------------------------------

                    assertion_results = []

                    # Assertion 1:
                    # HTTP status code
                    expected_status = (
                        201 if method == "POST" else 200
                    )

                    status_passed = (
                        response.status_code == expected_status
                    )

                    assertion_results.append({
                        "Check": "HTTP Status Code",
                        "Expected": expected_status,
                        "Actual": response.status_code,
                        "Result": (
                            "PASS" if status_passed else "FAIL"
                        ),
                    })

                    # Assertion 2:
                    # JSON Content-Type
                    json_passed = (
                        "application/json"
                        in content_type.lower()
                    )

                    assertion_results.append({
                        "Check": "JSON Content Type",
                        "Expected": "application/json",
                        "Actual": content_type or "Missing",
                        "Result": (
                            "PASS" if json_passed else "FAIL"
                        ),
                    })

                    # Assertion 3:
                    # Required JSON response fields
                    required_fields = [
                        "userId",
                        "id",
                        "title",
                        "body",
                    ]

                    fields_passed = (
                        isinstance(response_data, dict)
                        and all(
                            field in response_data
                            for field in required_fields
                        )
                    )

                    if isinstance(response_data, dict):

                        actual_fields = ", ".join(
                            field
                            for field in required_fields
                            if field in response_data
                        )

                    else:

                        actual_fields = (
                            "No valid JSON object"
                        )

                    assertion_results.append({
                        "Check": "Required JSON Fields",
                        "Expected": ", ".join(required_fields),
                        "Actual": actual_fields,
                        "Result": (
                            "PASS" if fields_passed else "FAIL"
                        ),
                    })

                    # ----------------------------------
                    # Overall Test Result
                    # ----------------------------------

                    overall_passed = all(
                        item["Result"] == "PASS"
                        for item in assertion_results
                    )

                    st.subheader(
                        "API Assertion Results"
                    )

                    st.dataframe(
                        pd.DataFrame(assertion_results),
                        width="stretch",
                    )

                    if overall_passed:

                        st.success(
                            "Overall API Test Result: PASS"
                        )

                    else:

                        st.error(
                            "Overall API Test Result: FAIL"
                        )

                    # ----------------------------------
                    # API Response Metrics
                    # ----------------------------------

                    st.subheader("API Response")

                    a1, a2, a3 = st.columns(3)

                    a1.metric(
                        "HTTP Status Code",
                        response.status_code,
                    )

                    a2.metric(
                        "Response Time",
                        f"{elapsed_ms:.2f} ms",
                    )

                    a3.metric(
                        "Response Size",
                        f"{len(response.content):,} bytes",
                    )

                    # HTTP response status
                    if 200 <= response.status_code < 300:

                        st.success(
                            "The API returned a successful HTTP status."
                        )

                    else:

                        st.warning(
                            "The API returned a non-success HTTP "
                            "status. Review the response details."
                        )

                    # Response headers
                    st.write("**Response Headers**")

                    st.json(
                        dict(response.headers)
                    )

                    # Response body
                    st.write("**Response Body**")

                    if response_data is not None:

                        st.json(response_data)

                    else:

                        st.code(
                            response.text[:10000]
                            or "(Empty response body)",
                            language="text",
                        )

                    # ----------------------------------
                    # Defect Intelligence
                    # ----------------------------------

                    st.subheader(
                        "Defect Intelligence"
                    )

                    failed_checks = [
                        item
                        for item in assertion_results
                        if item["Result"] == "FAIL"
                    ]

                    defect_report = []

                    # Generate defects from failed assertions
                    for index, defect in enumerate(
                        failed_checks,
                        start=1,
                    ):

                        # Severity classification
                        if defect["Check"] == "HTTP Status Code":

                            severity = (
                                "High"
                                if response.status_code >= 500
                                else "Medium"
                            )

                        elif defect["Check"] == "Required JSON Fields":

                            severity = "High"

                        else:

                            severity = "Medium"

                        defect_report.append({

                            "Defect ID": (
                                f"API-DEF-{index:03d}"
                            ),

                            "API Method": method,

                            "API Endpoint": url.strip(),

                            "Failed Check": defect["Check"],

                            "Expected": defect["Expected"],

                            "Actual": defect["Actual"],

                            "Severity": severity,

                            "Timestamp": pd.Timestamp.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                            "Response Time (ms)": round(
                                elapsed_ms,
                                2,
                            ),

                            "HTTP Status": response.status_code,

                        })

                    # ----------------------------------
                    # Display Defect Report
                    # ----------------------------------

                    if defect_report:

                        defect_df = pd.DataFrame(
                            defect_report
                        )

                        st.error(
                            f"Detected {len(defect_df)} "
                            "API defect(s)."
                        )

                        st.dataframe(
                            defect_df,
                            width="stretch",
                        )

                        # Download CSV report
                        defect_csv = defect_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Defect Report (CSV)",
                            data=defect_csv,
                            file_name=(
                                "insightx360_api_defects.csv"
                            ),
                            mime="text/csv",
                        )

                    else:

                        st.success(
                            "No API defects detected. "
                            "All assertions passed."
                        )

                # ----------------------------------
                # Error Handling
                # ----------------------------------

                except requests.exceptions.Timeout:

                    st.error(
                        "The API request timed out "
                        "after 10 seconds."
                    )

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"API request failed: {error}"
                    )
