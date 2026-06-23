# Bank Subscription Predictor

Predicts whether a client will subscribe to a term deposit based on bank marketing campaign data.

## Dataset Background
- **Source**: UCI Bank Marketing dataset (Portuguese bank direct marketing campaigns).
- **Records**: ~45,211, with 17 features.
- **Target**: `subscribed` – whether the client subscribed to a term deposit (`yes`/`no`).
- Features include demographics (`age`, `job`, `marital`, `education`), financials (`balance`, `default`, `housing`, `loan`), and campaign details (`duration`, `campaign`, `pdays`, `previous`, `poutcome`, etc.).

## Methodology and Approach
- **Data cleaning**: removes duplicates, maps binary columns to 0/1, encodes categorical variables with `LabelEncoder`, and standardises numerical features.
- **Exploratory analysis**: generates two figures –  
  - `dashboard_overview.png`: subscription distribution + correlation heatmap.  
  - `dashboard_distributions.png`: histograms of all numerical features.
- **Model**: Logistic Regression with `class_weight='balanced'` to handle class imbalance.
- **Evaluation**: accuracy, confusion matrix, and classification report on 20% test holdout.
- **Feature importance**: prints top 5 coefficients driving subscription (positive) and top 5 against (negative).
- **Inference**: provides a `predict_customer()` function to predict probability for a new client.

## Results
- **Accuracy**: ~89% (varies slightly with random seed).
- **Confusion matrix** and **classification report** displayed after training.
- **Key positive drivers**: longer `duration`, `poutcome=success`, more `previous` contacts.
- **Key negative drivers**: higher `campaign` count, higher `pdays`, slightly older `age`.

## Files

- 'bank_subscription_predictor.py' - main Python script
- 'bank_detailed.csv' - dataset
- 'dashboard_overview.png' - overview dashboard
- 'dashboard_distributions.png' - distributions dashboard
