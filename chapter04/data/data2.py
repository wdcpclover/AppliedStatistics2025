import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your dataset
df = pd.read_csv('data2/NHIS_Vision_and_Eye_Health_Surveillance_20240501.csv')

# Rename columns: replace spaces with underscores and convert to lowercase
df.columns = df.columns.str.replace(' ', '_').str.lower()

print(df.columns)

# 1. Age and Vision Health (ANOVA or Linear Regression)
model_age = ols('data_value ~ C(age)', data=df).fit()
print("ANOVA for Age and Vision Health:\n", model_age.summary())

# 2. Race/Ethnicity and Vision Health (Chi-squared Test or Logistic Regression)
contingency_race = pd.crosstab(df['raceethnicity'], df['data_value'])
chi2_race, p_race, _, _ = scipy.stats.chi2_contingency(contingency_race)
print(f"Chi-squared Test for Race/Ethnicity and Vision Health: Chi2 = {chi2_race}, P-value = {p_race}")

# 3. Gender and Specific Eye Conditions (Chi-squared Test or Logistic Regression)
model_gender = sm.formula.glm('data_value ~ C(gender)', family=sm.families.Binomial(), data=df).fit()
print("Logistic Regression for Gender and Eye Conditions:\n", model_gender.summary())

# 4. Impact of Risk Factors on Vision Health (Logistic Regression)
model_risk = sm.formula.glm('data_value ~ C(riskfactor)', family=sm.families.Binomial(), data=df).fit()
print("Logistic Regression for Impact of Risk Factors on Vision Health:\n", model_risk.summary())

# 5. Interaction Effects (Multivariate Regression or ANOVA)
model_interaction = ols('data_value ~ C(riskfactor) * C(age) * C(raceethnicity)', data=df).fit()
print("ANOVA for Interaction Effects:\n", model_interaction.summary())

# 6. Geographic Variation in Vision Health (Chi-squared Test or ANOVA)
if df['data_value'].dtype == 'object':  # Categorical data
    contingency_geo = pd.crosstab(df['locationdesc'], df['data_value'])
    chi2_geo, p_geo, _, _ = scipy.stats.chi2_contingency(contingency_geo)
    print(f"Chi-squared Test for Geographic Variation in Vision Health: Chi2 = {chi2_geo}, P-value = {p_geo}")
else:  # Continuous data
    model_geo = ols('data_value ~ C(locationdesc)', data=df).fit()
    print("ANOVA for Geographic Variation in Vision Health:\n", model_geo.summary())

# 7. Temporal Trends in Vision Health (Time Series Analysis or Linear Regression)
model_temporal = ols('data_value ~ yearstart', data=df).fit()
print("Linear Regression for Temporal Trends in Vision Health:\n", model_temporal.summary())

