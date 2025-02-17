import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your dataset
df = pd.read_csv('data1/heart_statlog_cleveland_hungary_final.csv')

# Rename columns: replace spaces with underscores and convert to lowercase
df.columns = df.columns.str.replace(' ', '_').str.lower()

print(df.columns)

# Chi-squared tests and ANOVA

# Sex and Heart Disease: Chi-squared Test
table_sex = pd.crosstab(df['sex'], df['target'])
chi2_sex, p_sex, _, _ = scipy.stats.chi2_contingency(table_sex)
print(f"Chi-squared test for Sex and Heart Disease: Chi2 = {chi2_sex}, P-value = {p_sex}")

# Type of Chest Pain and Heart Disease: Chi-squared Test
table_cp = pd.crosstab(df['chest_pain_type'], df['target'])
chi2_cp, p_cp, _, _ = scipy.stats.chi2_contingency(table_cp)
print(f"Chi-squared test for Chest Pain Type and Heart Disease: Chi2 = {chi2_cp}, P-value = {p_cp}")

# Age and Heart Disease: ANOVA
model_age = ols('age ~ C(target)', data=df).fit()
anova_age = sm.stats.anova_lm(model_age, typ=2)
print("ANOVA for Age and Heart Disease:\n", anova_age)

# Resting ECG Results and Heart Disease: Chi-squared Test
table_ecg = pd.crosstab(df['resting_ecg'], df['target'])
chi2_ecg, p_ecg, _, _ = scipy.stats.chi2_contingency(table_ecg)
print(f"Chi-squared test for Resting ECG and Heart Disease: Chi2 = {chi2_ecg}, P-value = {p_ecg}")

# Maximum Heart Rate and Heart Disease: ANOVA
model_hr = ols('max_heart_rate ~ C(target)', data=df).fit()
anova_hr = sm.stats.anova_lm(model_hr, typ=2)
print("ANOVA for Maximum Heart Rate and Heart Disease:\n", anova_hr)

# Exercise Induced Angina and Heart Disease: Chi-squared Test
table_angina = pd.crosstab(df['exercise_angina'], df['target'])
chi2_angina, p_angina, _, _ = scipy.stats.chi2_contingency(table_angina)
print(f"Chi-squared test for Exercise Induced Angina and Heart Disease: Chi2 = {chi2_angina}, P-value = {p_angina}")

# Fasting Blood Sugar and Heart Disease: Chi-squared Test
table_fbs = pd.crosstab(df['fasting_blood_sugar'], df['target'])
chi2_fbs, p_fbs, _, _ = scipy.stats.chi2_contingency(table_fbs)
print(f"Chi-squared test for Fasting Blood Sugar and Heart Disease: Chi2 = {chi2_fbs}, P-value = {p_fbs}")