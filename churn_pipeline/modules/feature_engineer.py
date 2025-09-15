import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for churn prediction with same logic as ChurnPipelineNoLeakage"""

    def fit(self, X, y=None):
        # Store column names for transforming
        self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy(), columns=self.input_features_)

        # Ensure numeric conversion
        numeric_cols_to_convert = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna({'TotalCharges': 0}, inplace=True)

        # Core risk features
        tenure = df.get('tenure', pd.Series(0)).replace(0, 1e-6)
        monthly_charges = df.get('MonthlyCharges', pd.Series(0)).replace(0, 1e-6)
        df['risk_score'] = monthly_charges / tenure
        df['tenure_monthly_ratio'] = tenure / monthly_charges
        df['low_tenure_high_risk'] = ((tenure < 12) & (df['risk_score'] > 10)).astype(int)

        # Service engagement
        service_cols = [col for col in df.columns if any(x in col.lower() for x in ['streaming', 'online', 'device', 'tech', 'backup'])]
        if service_cols:
            for col in service_cols:
                df[col] = pd.to_numeric(df[col].replace('No internet service', 0), errors='coerce')
            df[service_cols] = df[service_cols].fillna(0)
            df['service_engagement'] = df[service_cols].sum(axis=1)

        # Tenure/stability features
        df['tenure_stability'] = np.where(tenure > 24, 2, np.where(tenure > 12, 1, 0))
        contract_two_year = df.get('Contract_Two year', 0)
        df['high_stability'] = ((tenure > 18) | (contract_two_year == 1)).astype(int)
        df['stable_low_risk'] = ((tenure > 24) & (df['risk_score'] < 8)).astype(int)

        # Payment features
        risky_payment_check = df.get('PaymentMethod_Electronic check', 0)
        risky_payment_mail = df.get('PaymentMethod_Mailed check', 0)
        df['risky_payment'] = np.maximum(risky_payment_check, risky_payment_mail).astype(int)
        df['payment_tenure_adjusted'] = np.where((df['risky_payment'] == 1) & (tenure > 18), 0.5, df['risky_payment'])

        # New customer risk
        df['new_customer_risk'] = np.where(tenure <= 3, np.where(monthly_charges > 70, 1, 0.5), 0)

        # Engagement & internet features
        internet_no = df.get('InternetService_No', 0)
        if 'service_engagement' in df.columns:
            df['engagement_no_internet'] = ((df['service_engagement'] >= 4) & (internet_no == 1)).astype(int)
            df['truly_engaged'] = ((df['service_engagement'] >= 3) & (internet_no == 0)).astype(int)
        fiber_optic = df.get('InternetService_Fiber optic', 0)
        df['fiber_established'] = ((fiber_optic == 1) & (tenure > 8)).astype(int)

        # Family stability
        partner_yes = df.get('Partner_Yes', 0)
        dependents_yes = df.get('Dependents_Yes', 0)
        df['family_stability'] = np.maximum(partner_yes, dependents_yes).astype(int)

        # Risk category
        df['risk_category'] = np.where(df['risk_score'] > 25, 3,
                                  np.where(df['risk_score'] > 15, 2,
                                           np.where(df['risk_score'] > 8, 1, 0)))

        # Aggregate stability & warning scores
        stability_features = [f for f in ['stable_low_risk', 'high_stability', 'fiber_established', 'truly_engaged', 'family_stability'] if f in df.columns]
        if stability_features:
            df['churn_immunity'] = df[stability_features].astype(float).sum(axis=1)

        warning_features = [f for f in ['new_customer_risk', 'risky_payment', 'low_tenure_high_risk', 'engagement_no_internet'] if f in df.columns]
        if warning_features:
            df['early_warning_score'] = df[warning_features].astype(float).sum(axis=1)

        return df
