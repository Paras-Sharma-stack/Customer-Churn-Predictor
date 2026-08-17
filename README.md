# Customer Churn Predicton Engine

## Project Overview
The **Customer Churn Predictor** is an end-to-end Machine Learning pipeline designed to predict whether a customer is likely to cancel their subscription. By analyzing historical usage behaviors, demographic data, and account billing details, this custom ML engine identifies at-risk customers, allowing businesses to take proactive retention measures.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForestClassifier)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## Dataset
This project utilizes the classic **Telco Customer Churn** dataset (available via Kaggle). It contains over 7,000 customer profiles with features such as:
* **Demographics:** Gender, Senior Citizen status, Partners, Dependents.
* **Account Information:** Tenure, Contract type, Payment method, Monthly/Total charges.
* **Services Subscribed:** Internet, Phone, Streaming TV/Movies, Tech Support.
* **Target Variable:** `Churn` (Yes/No)

* **Link:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Key Features & Workflow
1. **Data Cleaning & Preprocessing:** 
   * Handled hidden null values and corrected data type mismatches (e.g., converting `TotalCharges` to numeric).
   * Implemented One-Hot Encoding (`pd.get_dummies`) to translate categorical text into mathematically optimized features while avoiding the dummy variable trap.
2. **Exploratory Data Analysis (EDA):**
   * Visualized target class imbalances using Seaborn count plots.
   * Mapped out churn distributions against key risk factors like user tenure and contract types.
3. **Machine Learning Model:**
   * Structured data using an 80/20 Train/Test split for optimal learning and validation.
   * Trained a `RandomForestClassifier` to map out complex, non-linear relationships in customer behavior.
4. **Model Evaluation:**
   * Measured predictive performance using overall Accuracy scores.
   * Generated a Confusion Matrix to track exact False Positive (false alarms) and False Negative (missed opportunities) rates.

## Future Roadmap

While the current pipeline successfully identifies churn risk among **existing** customers with established tenure, the next phases of this project will transform it into an intelligent, proactive business tool:

### 1. "Cold-Start" New Customer Prediction Model
The current Random Forest model relies heavily on historical data like `tenure` and `TotalCharges`. 
* **The Plan:** Train a secondary classification model strictly optimized for newly onboarded customers (Tenure = 0-1 months). This model will predict early churn risk based entirely on initial setup choices (e.g., Contract Type, Demographics, Chosen Services), allowing the business to intervene before a customer establishes a usage history.

### 2. LLM Integration & AI Agents
Raw probability scores are useful, but business teams need actionable intelligence. 
* **The Plan:** Integrate a Large Language Model (LLM) to process the output of the predictive engine. Instead of just outputting a `1` or `0`, the LLM agent will:
  * Automatically generate plain-text **Key Insights** explaining *why* a specific user was flagged (e.g., *"This user is high-risk due to a combination of a Month-to-Month contract and high monthly fees without Tech Support"*).
  * Provide dynamic, customized **Retention Suggestions** for the marketing team (e.g., *"Offer a 15% discount to upgrade to a 1-year contract"*).

## Installation & Usage
1. Clone this repository to your local machine.
2. Ensure your virtual environment is active.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
