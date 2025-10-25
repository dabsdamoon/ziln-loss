import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load the data
print("Loading data...")
df = pd.read_excel('data/online_retail_II.xlsx')
print(f"Data loaded: {len(df):,} transactions")

# Calculate transaction value
df['TransactionValue'] = df['Quantity'] * df['Price']

# Calculate LTV per customer (total transaction value across all purchases)
customer_ltv = df.groupby('Customer ID').agg({
    'TransactionValue': 'sum',
    'Invoice': 'nunique',  # Number of unique invoices
    'Quantity': 'sum'
}).reset_index()

customer_ltv.columns = ['CustomerID', 'LTV', 'NumInvoices', 'TotalQuantity']

# Remove any customers with missing IDs
customer_ltv = customer_ltv[customer_ltv['CustomerID'].notna()]

# Sort by LTV
customer_ltv = customer_ltv.sort_values('LTV', ascending=False).reset_index(drop=True)

print(f"\nTotal customers: {len(customer_ltv):,}")
print(f"LTV range: ${customer_ltv['LTV'].min():,.2f} to ${customer_ltv['LTV'].max():,.2f}")
print(f"Negative LTVs (net returns): {(customer_ltv['LTV'] < 0).sum():,} customers")

import ipdb; ipdb.set_trace()

# Calculate Pareto statistics
total_ltv = customer_ltv['LTV'].sum()
customer_ltv['CumulativeValue'] = customer_ltv['LTV'].cumsum()
customer_ltv['CumulativePct'] = customer_ltv['CumulativeValue'] / total_ltv * 100

# Find top 1%, 5%, 10%, 20%
top_1_pct_value = customer_ltv.head(int(len(customer_ltv) * 0.01))['LTV'].sum()
top_5_pct_value = customer_ltv.head(int(len(customer_ltv) * 0.05))['LTV'].sum()
top_10_pct_value = customer_ltv.head(int(len(customer_ltv) * 0.10))['LTV'].sum()
top_20_pct_value = customer_ltv.head(int(len(customer_ltv) * 0.20))['LTV'].sum()

print(f"\nValue Concentration (Pareto Analysis):")
print(f"  Top 1% of customers generate: {top_1_pct_value/total_ltv*100:.1f}% of total value")
print(f"  Top 5% of customers generate: {top_5_pct_value/total_ltv*100:.1f}% of total value")
print(f"  Top 10% of customers generate: {top_10_pct_value/total_ltv*100:.1f}% of total value")
print(f"  Top 20% of customers generate: {top_20_pct_value/total_ltv*100:.1f}% of total value")

# Calculate skewness and kurtosis
print(f"\nDistribution Statistics:")
print(f"  Mean LTV: ${customer_ltv['LTV'].mean():,.2f}")
print(f"  Median LTV: ${customer_ltv['LTV'].median():,.2f}")
print(f"  Std Dev: ${customer_ltv['LTV'].std():,.2f}")
print(f"  Skewness: {stats.skew(customer_ltv['LTV']):.2f} (highly right-skewed)")
print(f"  Kurtosis: {stats.kurtosis(customer_ltv['LTV']):.2f} (heavy tails)")

# Create the visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Heavy-Tailed LTV Distribution Analysis\nOnline Retail II Dataset',
             fontsize=16, fontweight='bold', y=0.995)

# 1. Full distribution histogram (with outliers)
ax1 = axes[0, 0]
ax1.hist(customer_ltv['LTV'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Customer Lifetime Value ($)', fontsize=10)
ax1.set_ylabel('Number of Customers', fontsize=10)
ax1.set_title('LTV Distribution (Full Range)\nShowing extreme heavy tail', fontsize=11, fontweight='bold')
ax1.axvline(customer_ltv['LTV'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${customer_ltv["LTV"].mean():,.0f}')
ax1.axvline(customer_ltv['LTV'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${customer_ltv["LTV"].median():,.0f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Log-scale histogram (better view of tail)
ax2 = axes[0, 1]
# Filter out negative values for log scale
positive_ltv = customer_ltv[customer_ltv['LTV'] > 0]['LTV']
ax2.hist(np.log10(positive_ltv), bins=50, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Log10(Customer Lifetime Value)', fontsize=10)
ax2.set_ylabel('Number of Customers', fontsize=10)
ax2.set_title('LTV Distribution (Log Scale)\nRevealing the heavy tail structure', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Pareto chart (cumulative value by customer rank)
ax3 = axes[0, 2]
customer_pct = np.arange(1, len(customer_ltv) + 1) / len(customer_ltv) * 100
ax3.plot(customer_pct, customer_ltv['CumulativePct'], color='darkgreen', linewidth=2)
ax3.axhline(80, color='red', linestyle='--', alpha=0.7, label='80% of value')
ax3.axvline(20, color='red', linestyle='--', alpha=0.7, label='~20% of customers')
ax3.set_xlabel('Cumulative % of Customers (Ranked by LTV)', fontsize=10)
ax3.set_ylabel('Cumulative % of Total Value', fontsize=10)
ax3.set_title('Pareto Curve: Value Concentration\nShowing 80-20 principle', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)

# 4. Top 100 customers bar chart
ax4 = axes[1, 0]
top_100 = customer_ltv.head(100)
ax4.bar(range(len(top_100)), top_100['LTV'], color='steelblue', alpha=0.7)
ax4.set_xlabel('Customer Rank', fontsize=10)
ax4.set_ylabel('Lifetime Value ($)', fontsize=10)
ax4.set_title('Top 100 Customers by LTV\nShowing massive disparity', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Box plot comparing segments
ax5 = axes[1, 1]
# Create segments
segments = []
labels = []
segments.append(customer_ltv.head(int(len(customer_ltv) * 0.01))['LTV'])
labels.append('Top 1%')
segments.append(customer_ltv.iloc[int(len(customer_ltv) * 0.01):int(len(customer_ltv) * 0.10)]['LTV'])
labels.append('1-10%')
segments.append(customer_ltv.iloc[int(len(customer_ltv) * 0.10):int(len(customer_ltv) * 0.50)]['LTV'])
labels.append('10-50%')
segments.append(customer_ltv.tail(int(len(customer_ltv) * 0.50))['LTV'])
labels.append('Bottom 50%')

ax5.boxplot(segments, labels=labels, showfliers=False)
ax5.set_ylabel('Lifetime Value ($)', fontsize=10)
ax5.set_title('LTV Distribution by Customer Segment\nShowing extreme differences', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Distribution of negative LTVs (returns/cancellations)
ax6 = axes[1, 2]
negative_ltv = customer_ltv[customer_ltv['LTV'] < 0]['LTV']
positive_ltv_for_hist = customer_ltv[customer_ltv['LTV'] > 0]['LTV']
ax6.hist([positive_ltv_for_hist, negative_ltv], bins=50,
         label=['Positive LTV', f'Negative LTV (n={len(negative_ltv)})'],
         color=['green', 'red'], alpha=0.6, edgecolor='black')
ax6.set_xlabel('Lifetime Value ($)', fontsize=10)
ax6.set_ylabel('Number of Customers', fontsize=10)
ax6.set_title('Positive vs Negative LTV\nImpact of cancellations', fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim(-10000, 50000)  # Focus on main distribution

plt.tight_layout()
plt.savefig('ltv_heavy_tail_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'ltv_heavy_tail_analysis.png'")
plt.show()

# Additional statistical summary
print("\n" + "="*60)
print("SUMMARY: Heavy-Tailed LTV Distribution Characteristics")
print("="*60)
print(f"1. EXTREME SKEWNESS: The distribution is heavily right-skewed")
print(f"   - Mean (${customer_ltv['LTV'].mean():,.0f}) >> Median (${customer_ltv['LTV'].median():,.0f})")
print(f"   - Skewness coefficient: {stats.skew(customer_ltv['LTV']):.2f}")
print(f"\n2. VALUE CONCENTRATION: A small number of customers drive most value")
print(f"   - Top 1% contribute {top_1_pct_value/total_ltv*100:.1f}% of total value")
print(f"   - Top 10% contribute {top_10_pct_value/total_ltv*100:.1f}% of total value")
print(f"\n3. LONG TAIL: Many low-value customers in the tail")
print(f"   - Bottom 50% contribute only {(1 - top_20_pct_value/total_ltv)*100:.1f}% of value")
print(f"\n4. CANCELLATIONS: Negative LTVs stretch the distribution")
print(f"   - {len(negative_ltv)} customers ({len(negative_ltv)/len(customer_ltv)*100:.1f}%) have negative LTV")
print(f"   - These represent net returners/cancellers")
print("="*60)
