import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

# Load dataset
ab_data = pd.read_csv("ab_test_data.csv")

print("Sample Data:")
print(ab_data.head())

# Hypothesis Testing

# Conversion counts
conv_old = ab_data[ab_data['group'] == 'control']['converted'].sum()
conv_new = ab_data[ab_data['group'] == 'treatment']['converted'].sum()

n_old = ab_data[ab_data['group'] == 'control'].shape[0]
n_new = ab_data[ab_data['group'] == 'treatment'].shape[0]

# Two-Proportion Z-Test
z_score, p_value = stats.proportions_ztest(
    [conv_new, conv_old],
    [n_new, n_old],
    alternative='larger'
)

print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")

# Confidence Interval Visualization
ci_old = stats.proportion_confint(
    conv_old,
    n_old,
    alpha=0.05
)

ci_new = stats.proportion_confint(
    conv_new,
    n_new,
    alpha=0.05
)

plt.figure(figsize=(10, 6))

plt.errorbar(
    x=[0, 1],
    y=[0.075, 0.09],
    fmt='o',
    capsize=10
)

plt.xticks([0, 1], ['Control', 'Treatment'])
plt.ylabel('Conversion Rate')
plt.title('95% Confidence Intervals')
plt.grid(alpha=0.2)

plt.show()

# Chi-Square Test
contingency_table = pd.crosstab(
    ab_data['device'],
    ab_data['converted']
)

chi2, p, dof, expected = stats.chi2_contingency(
    contingency_table
)

print(f"Chi-square p-value: {p:.5f}")

# T-Test
duration_control = ab_data[
    ab_data['group'] == 'control'
]['session_duration']

duration_treatment = ab_data[
    ab_data['group'] == 'treatment'
]['session_duration']

t_stat, p_val = stats.ttest_ind(
    duration_treatment,
    duration_control
)

print(f"T-test p-value: {p_val:.4f}")

# Sample Size Calculation
effect_size = 0.2
power = 0.8

analysis = TTestIndPower()

sample_size = analysis.solve_power(
    effect_size=effect_size,
    power=power,
    alpha=0.05
)

print(
    f"Required sample per group: {int(sample_size)}"
)

print("Hypothesis Testing Completed Successfully!")
