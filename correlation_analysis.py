#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = '/Users/varnithakurli/Library/CloudStorage/OneDrive-UCB-O365/AI Project-Varni and Garrett/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)
df = df[df['Coarse_Topic'] != "Poverty & Inequality"]


# In[5]:


output_dir = Path('/Users/varnithakurli/Library/CloudStorage/OneDrive-UCB-O365/AI Project-Varni and Garrett/Varni_analysis/1_data/')
output_dir.mkdir(parents=True, exist_ok=True)                
                 
                 
# Get year columns
year_cols = [col for col in df.columns if col.startswith('year_')]
print(f"Found {len(year_cols)} year columns")

# Get unique coarse topics
coarse_topics = df['Coarse_Topic'].dropna().unique()
print(f"Found {len(coarse_topics)} unique coarse topics")

# Dictionary to store removed indicators by topic
removed_indicators = {}
# Dictionary to store retained indicators by topic
retained_indicators = {}

# Correlation threshold for "highly correlated" indicators
CORR_THRESHOLD = 0.7

# Create a summary table for correlation
correlation_summary = []

# Process each coarse topic
for topic in coarse_topics:
    print(f"\nProcessing topic: {topic}")
    
    # Filter data for this coarse topic
    topic_df = df[df['Coarse_Topic'] == topic].copy()
    
    # Get unique indicators for this topic
    indicators = topic_df['Indicator Name'].unique()
    
    if len(indicators) < 2:
        print(f"Topic {topic} has fewer than 2 indicators, skipping")
        correlation_summary.append({
            'Coarse_Topic': topic,
            'Original_Indicator_Count': len(indicators),
            'Highly_Correlated_Pairs': 0,
            'Indicators_Removed': 0,
            'Indicators_Retained': len(indicators)
        })
        removed_indicators[topic] = []
        retained_indicators[topic] = list(indicators)
        continue
    
    # Create correlation data
    corr_data = pd.DataFrame()
    indicator_to_code = {}
    
    for indicator in indicators:
        indicator_df = topic_df[topic_df['Indicator Name'] == indicator]
        indicator_to_code[indicator] = indicator_df['Indicator Code'].iloc[0]
        
        # Melt to convert from wide to long format
        melted = pd.melt(
            indicator_df,
            id_vars=['Country Code'], 
            value_vars=year_cols,
            var_name='Year', 
            value_name=indicator
        )
        
        melted['Year'] = melted['Year'].str.replace('year_', '')
        
        # Add to correlation data
        if corr_data.empty:
            corr_data = melted[['Country Code', 'Year', indicator]]
        else:
            corr_data = pd.merge(
                corr_data, 
                melted[['Country Code', 'Year', indicator]], 
                on=['Country Code', 'Year'], 
                how='outer'
            )
    
    # Calculate correlation
    correlation = corr_data.drop(['Country Code', 'Year'], axis=1).corr()
    
    # Remove NaN columns/rows
    correlation = correlation.dropna(how='all').dropna(how='all', axis=1)
    
    if correlation.shape[0] < 2:
        print(f"Insufficient data for correlation analysis after removing NaNs for topic {topic}")
        correlation_summary.append({
            'Coarse_Topic': topic,
            'Original_Indicator_Count': len(indicators),
            'Highly_Correlated_Pairs': 0,
            'Indicators_Removed': 0,
            'Indicators_Retained': len(indicators)
        })
        removed_indicators[topic] = []
        retained_indicators[topic] = list(indicators)
        continue
    
    # Create a list of indicator pairs with their correlation values
    corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            ind1 = correlation.columns[i]
            ind2 = correlation.columns[j]
            corr_val = correlation.iloc[i, j]
            
            if pd.notna(corr_val):
                abs_corr = abs(corr_val)
                corr_pairs.append((ind1, ind2, abs_corr))
    
    # Sort pairs by correlation value (descending)
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Identify highly correlated pairs
    high_corr_pairs = [pair for pair in corr_pairs if pair[2] >= CORR_THRESHOLD]
    
    # Create a table of all correlation pairs
    corr_table = pd.DataFrame(corr_pairs, columns=['Indicator1', 'Indicator2', 'Absolute_Correlation'])
    corr_table['Indicator1_Code'] = corr_table['Indicator1'].map(indicator_to_code)
    corr_table['Indicator2_Code'] = corr_table['Indicator2'].map(indicator_to_code)
    corr_table['Is_Highly_Correlated'] = corr_table['Absolute_Correlation'] >= CORR_THRESHOLD
    
    # Save correlation table for this topic
    corr_table_file = output_dir / f"{topic.replace(' ', '_')}_correlation_table.csv"
    corr_table.to_csv(corr_table_file, index=False)
    print(f"Saved correlation table to {corr_table_file}")
    
    # Algorithm to remove highly correlated indicators
    # We'll use a greedy approach: remove indicators that appear most frequently in high correlation pairs
    to_remove = set()
    indicator_frequency = {}
    
    # Count how many times each indicator appears in high correlation pairs
    for ind1, ind2, corr_val in high_corr_pairs:
        indicator_frequency[ind1] = indicator_frequency.get(ind1, 0) + 1
        indicator_frequency[ind2] = indicator_frequency.get(ind2, 0) + 1
    
    # Sort indicators by frequency in high correlation pairs
    sorted_indicators = sorted(indicator_frequency.items(), key=lambda x: x[1], reverse=True)
    
    # Iteratively remove indicators until no high correlations remain
    for ind, freq in sorted_indicators:
        # Skip if this indicator is already marked for removal
        if ind in to_remove:
            continue
        
        # Check if there are still high correlations with this indicator
        remaining_high_corr = False
        for pair in high_corr_pairs:
            ind1, ind2, corr_val = pair
            if (ind1 == ind and ind2 not in to_remove) or (ind2 == ind and ind1 not in to_remove):
                remaining_high_corr = True
                break
        
        # If no remaining high correlations, we don't need to remove this indicator
        if not remaining_high_corr:
            continue
        
        # Otherwise, mark this indicator for removal
        to_remove.add(ind)
        
        # Check if we've eliminated all high correlations
        all_handled = True
        for pair in high_corr_pairs:
            ind1, ind2, corr_val = pair
            if ind1 not in to_remove and ind2 not in to_remove:
                all_handled = False
                break
        
        if all_handled:
            break
    
    # Store the results
    removed_indicators[topic] = list(to_remove)
    retained_indicators[topic] = [ind for ind in correlation.columns if ind not in to_remove]
    
    # Add to summary
    correlation_summary.append({
        'Coarse_Topic': topic,
        'Original_Indicator_Count': len(correlation.columns),
        'Highly_Correlated_Pairs': len(high_corr_pairs),
        'Indicators_Removed': len(to_remove),
        'Indicators_Retained': len(correlation.columns) - len(to_remove)
    })
    
    # Create visualization of correlation matrix with labeled indicators to remove
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                annot=True, fmt=".2f", annot_kws={"size": 8})
    
    # Mark indicators to be removed
    for i, ind in enumerate(correlation.columns):
        if ind in to_remove:
            plt.text(i + 0.5, i, "X", fontsize=12, color="black", 
                    ha="center", va="center")
    
    plt.title(f"Correlation Matrix for {topic}\nX = Indicators to be removed")
    plt.tight_layout()
    
    # Save correlation heatmap
    vis_file = output_dir / f"{topic.replace(' ', '_')}_correlation_heatmap.png"
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {vis_file}")

# Create summary table
summary_df = pd.DataFrame(correlation_summary)
summary_file = output_dir / "correlation_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Saved correlation summary to {summary_file}")

# Create detailed report
report_file = output_dir / "indicator_correlation_report.txt"
with open(report_file, 'w') as f:
    f.write("INDICATOR CORRELATION ANALYSIS\n")
    f.write("=============================\n\n")
    
    f.write("SUMMARY\n")
    f.write("-------\n")
    f.write(f"Total topics analyzed: {len(coarse_topics)}\n")
    f.write(f"Correlation threshold for removal: {CORR_THRESHOLD}\n\n")
    
    for row in summary_df.sort_values('Highly_Correlated_Pairs', ascending=False).itertuples():
        f.write(f"Topic: {row.Coarse_Topic}\n")
        f.write(f"  Original indicators: {row.Original_Indicator_Count}\n")
        f.write(f"  Highly correlated pairs: {row.Highly_Correlated_Pairs}\n")
        f.write(f"  Indicators removed: {row.Indicators_Removed}\n")
        f.write(f"  Indicators retained: {row.Indicators_Retained}\n\n")
    
    f.write("\nDETAILED RESULTS BY TOPIC\n")
    f.write("========================\n\n")
    
    for topic in coarse_topics:
        f.write(f"\nTOPIC: {topic}\n")
        f.write("=" * (len(topic) + 7) + "\n")
        
        # Get removed and retained indicators
        removed = removed_indicators[topic]
        retained = retained_indicators[topic]
        
        f.write(f"Indicators removed ({len(removed)}):\n")
        if removed:
            for i, ind in enumerate(removed, 1):
                code = indicator_to_code.get(ind, "Unknown")
                f.write(f"{i}. {ind} [{code}]\n")
        else:
            f.write("None\n")
        
        f.write(f"\nIndicators retained ({len(retained)}):\n")
        if retained:
            for i, ind in enumerate(retained, 1):
                code = indicator_to_code.get(ind, "Unknown")
                f.write(f"{i}. {ind} [{code}]\n")
        else:
            f.write("None\n")
        
        f.write("\n" + "-" * 50 + "\n")

print(f"Saved detailed report to {report_file}")

# Create a visualization of the number of indicators removed vs. retained by topic
summary_vis_df = summary_df.sort_values('Original_Indicator_Count', ascending=False).head(15)

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
summary_vis_df['Indicators_Removed_Pct'] = summary_vis_df['Indicators_Removed'] / summary_vis_df['Original_Indicator_Count'] * 100

# Bar chart of indicator counts
ax = summary_vis_df.plot(
    kind='bar', 
    x='Coarse_Topic', 
    y=['Indicators_Retained', 'Indicators_Removed'],
    stacked=True, 
    color=['green', 'red'],
    figsize=(14, 10)
)
plt.title('Indicators Retained vs. Removed by Coarse Topic (Top 15 by Indicator Count)')
plt.xticks(rotation=45, ha='right')
plt.legend(['Retained', 'Removed'])
plt.ylabel('Number of Indicators')
plt.tight_layout()

# Percentage of indicators removed
plt.subplot(2, 1, 2)
plt.bar(summary_vis_df['Coarse_Topic'], summary_vis_df['Indicators_Removed_Pct'])
plt.title('Percentage of Indicators Removed by Coarse Topic')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.tight_layout()

summary_vis_file = output_dir / "indicator_removal_summary.png"
plt.savefig(summary_vis_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved indicator removal summary visualization to {summary_vis_file}")

# Create a filtered dataset with only retained indicators
print("Creating filtered dataset with only retained indicators...")
filtered_df = pd.DataFrame()

for topic in coarse_topics:
    topic_retained = retained_indicators[topic]
    
    if not topic_retained:
        continue
    
    for indicator in topic_retained:
        indicator_df = df[(df['Coarse_Topic'] == topic) & (df['Indicator Name'] == indicator)]
        filtered_df = pd.concat([filtered_df, indicator_df])

# Save filtered dataset
filtered_file = output_dir / "filtered_dataset_without_correlated_indicators.csv"
filtered_df.to_csv(filtered_file, index=False)
print(f"Saved filtered dataset to {filtered_file}")

print("\nAnalysis complete!")


# In[ ]:


# add target variables to the filtered dataset

