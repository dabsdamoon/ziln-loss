"""
EDA: Analyze target variable distribution for heavy tail and right-skewness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load training data
print("Loading training data...")
train = pd.read_parquet('data/processed/train.parquet')

target = train['future_12m_purchase_value']

print(f"\nTarget Variable Statistics:")
print(f"=" * 60)
print(f"Count: {len(target):,}")
print(f"Mean: ${target.mean():.2f}")
print(f"Median: ${target.median():.2f}")
print(f"Std Dev: ${target.std():.2f}")
print(f"Min: ${target.min():.2f}")
print(f"Max: ${target.max():.2f}")
print(f"\nSkewness: {target.skew():.4f}")
print(f"Kurtosis: {target.kurtosis():.4f}")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: ${target.quantile(p/100):.2f}")

# Calculate coefficient of variation
cv = (target.std() / target.mean()) * 100
print(f"\nCoefficient of Variation: {cv:.2f}%")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Histogram with KDE
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(target, bins=100, alpha=0.6, color='skyblue', edgecolor='black', density=True)
target.plot(kind='kde', ax=ax1, color='red', linewidth=2)
ax1.axvline(target.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: ${target.mean():.2f}')
ax1.axvline(target.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${target.median():.2f}')
ax1.set_xlabel('12-Month Purchase Value ($)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Distribution of Target Variable (12-Month Purchase Value)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Log-scale histogram to better see the tail
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(target, bins=100, alpha=0.7, color='coral', edgecolor='black')
ax2.set_yscale('log')
ax2.set_xlabel('Purchase Value ($)', fontsize=10)
ax2.set_ylabel('Frequency (log scale)', fontsize=10)
ax2.set_title('Histogram (Log Scale)\nShowing Heavy Tail', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Box plot
ax3 = fig.add_subplot(gs[1, 0])
bp = ax3.boxplot(target, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))
ax3.set_ylabel('Purchase Value ($)', fontsize=10)
ax3.set_title('Box Plot\n(Outliers indicate heavy tail)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Q-Q plot (to check normality)
ax4 = fig.add_subplot(gs[1, 1])
stats.probplot(target, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot vs Normal Distribution\n(Deviation shows skewness)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. ECDF (Empirical Cumulative Distribution)
ax5 = fig.add_subplot(gs[1, 2])
sorted_target = np.sort(target)
ecdf = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
ax5.plot(sorted_target, ecdf, linewidth=2, color='purple')
ax5.set_xlabel('Purchase Value ($)', fontsize=10)
ax5.set_ylabel('Cumulative Probability', fontsize=10)
ax5.set_title('ECDF\n(Tail behavior)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80th percentile')
ax5.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
ax5.legend(fontsize=8)

# 6. Distribution of log-transformed target
ax6 = fig.add_subplot(gs[2, 0])
# Add small constant to avoid log(0)
log_target = np.log1p(target)
ax6.hist(log_target, bins=100, alpha=0.7, color='green', edgecolor='black', density=True)
log_target.plot(kind='kde', ax=ax6, color='darkgreen', linewidth=2)
ax6.set_xlabel('log(1 + Purchase Value)', fontsize=10)
ax6.set_ylabel('Density', fontsize=10)
ax6.set_title('Log-Transformed Distribution\n(More symmetric if right-skewed)', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Violin plot
ax7 = fig.add_subplot(gs[2, 1])
parts = ax7.violinplot([target], vert=True, showmeans=True, showmedians=True)
ax7.set_ylabel('Purchase Value ($)', fontsize=10)
ax7.set_title('Violin Plot\n(Shape shows distribution)', fontsize=11, fontweight='bold')
ax7.set_xticks([1])
ax7.set_xticklabels(['Target'])
ax7.grid(True, alpha=0.3, axis='y')

# 8. Tail analysis - Top 10% vs rest
ax8 = fig.add_subplot(gs[2, 2])
percentile_90 = target.quantile(0.9)
bottom_90 = target[target <= percentile_90]
top_10 = target[target > percentile_90]

data_to_plot = [bottom_90, top_10]
labels = [f'Bottom 90%\n(n={len(bottom_90):,})\nMean=${bottom_90.mean():.2f}',
          f'Top 10%\n(n={len(top_10):,})\nMean=${top_10.mean():.2f}']

positions = [1, 2]
bp = ax8.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True,
                  boxprops=dict(facecolor='lightgreen', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
ax8.set_ylabel('Purchase Value ($)', fontsize=10)
ax8.set_title('Heavy Tail Analysis\nBottom 90% vs Top 10%', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
ax8.tick_params(axis='x', labelsize=8)

# Add overall title
fig.suptitle('Target Variable Distribution Analysis: Heavy Tail & Right-Skewness Check',
             fontsize=16, fontweight='bold', y=0.995)

# Add text box with key metrics
textstr = f'''Key Indicators of Heavy Tail & Right-Skewness:
• Skewness: {target.skew():.3f} (>0 = right-skewed)
• Kurtosis: {target.kurtosis():.3f} (>3 = heavy tail)
• Mean > Median: ${target.mean():.2f} > ${target.median():.2f} ✓
• CV: {cv:.1f}% (high variability)
• Top 10% accounts for {(top_10.sum()/target.sum()*100):.1f}% of total value'''

fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('target_distribution_heavy_tail_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print("Plot saved as 'target_distribution_heavy_tail_analysis.png'")
print(f"{'='*60}")

# Additional heavy tail analysis
print(f"\nHeavy Tail Analysis:")
print(f"=" * 60)
print(f"Top 10% of customers account for {(top_10.sum()/target.sum()*100):.2f}% of total purchase value")
print(f"Top 5% of customers account for {(target[target > target.quantile(0.95)].sum()/target.sum()*100):.2f}% of total purchase value")
print(f"Top 1% of customers account for {(target[target > target.quantile(0.99)].sum()/target.sum()*100):.2f}% of total purchase value")

print(f"\nConclusion:")
print(f"=" * 60)
if target.skew() > 1:
    print("✓ STRONG right-skewness detected (skewness > 1)")
elif target.skew() > 0.5:
    print("✓ MODERATE right-skewness detected (skewness > 0.5)")
else:
    print("✗ Weak or no right-skewness")

if target.kurtosis() > 3:
    print("✓ HEAVY tail detected (excess kurtosis > 0)")
else:
    print("✗ No heavy tail detected")

if target.mean() > target.median():
    print("✓ Mean > Median confirms right-skewness")

plt.close()
