#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pathlib import Path
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Merge indicator time-series data with Categorization data

# In[2]:


# Load the WDI CSV data
df = pd.read_csv('/Users/varnithakurli/Library/CloudStorage/OneDrive-UCB-O365/AI Project-Varni and Garrett/Data/WDI_CSV_2025_01_28/WDICSV.csv')
print("Original column names:", df.columns.tolist())

# Rename year columns to year_XXXX format
year_cols = [str(year) for year in range(1960, 2024)]
rename_dict = {col: f'year_{col}' for col in year_cols if col in df.columns}
df = df.rename(columns=rename_dict)
print("New column names:", df.columns.tolist())

#Load Categorization data 
df1 = pd.read_csv('/Users/varnithakurli/Library/CloudStorage/OneDrive-UCB-O365/AI Project-Varni and Garrett/Data/WDI_CSV_2025_01_28/WDISeries.csv')
print("Original column names:", df1.columns.tolist())

# Rename 'Series Code' to 'Indicator Code'
df1 = df1.rename(columns={'Series Code': 'Indicator Code'})
print("Modified df1 column names:", df1.columns.tolist())

# Merge df and df1 on 'Indicator Code'
merged_df = pd.merge(df, df1[['Indicator Code', 'Topic']], on='Indicator Code', how='left')
print("Merged dataframe columns:", merged_df.columns.tolist())

# Check the first few rows of the merged dataframe
print("\nFirst 5 rows of merged dataframe:")
print(merged_df.head())


# # How far back should we go with the indicators? 

# In[3]:


# Calculate data availability for each year column in the merged dataframe
years = []
data_counts = []

# Process each year column (1960-2023)
for year in range(1960, 2024):
    year_col = f'year_{year}'
    if year_col in merged_df.columns:
        # Count non-null values
        count = merged_df[year_col].notna().sum()
        years.append(year)
        data_counts.append(count)

# Create a basic plot
plt.figure(figsize=(12, 6))
plt.bar(years, data_counts, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Available Data Points')
plt.title('Data Availability Over Time (Merged Dataset)')
plt.xticks(np.arange(1960, 2024, 5), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Shade different time periods
plt.axvspan(2000, 2020, alpha=0.2, color='green', label='High Coverage (2000-2020)')
plt.axvspan(1990, 2000, alpha=0.2, color='yellow', label='Good Coverage (1990-2000)')
plt.axvspan(1970, 1990, alpha=0.2, color='orange', label='Moderate Coverage (1970-1990)')
plt.legend()
plt.tight_layout()
plt.savefig('merged_data_availability.png')
plt.show()

# Group by decade for summary
decade_data = {}
for i, year in enumerate(years):
    decade = (year // 10) * 10
    if decade not in decade_data:
        decade_data[decade] = []
    decade_data[decade].append(data_counts[i])

# Print decade summary
print("\nData Availability by Decade (Merged Dataset):")
for decade, counts in sorted(decade_data.items()):
    avg_count = sum(counts) / len(counts)
    min_count = min(counts)
    max_count = max(counts)
    print(f"{decade}s: Avg={avg_count:.1f}, Min={min_count}, Max={max_count}")

# Find best decade
best_decade = max(decade_data.items(), key=lambda x: sum(x[1])/len(x[1]))[0]
print("\nRecommendation for Merged Dataset:")
print(f"Based on data availability, focus on data from {best_decade} onwards.")
print("The period 2000-2020 offers the most comprehensive coverage.")


# # Topic Variable -Cleaning and Missing values- On trimmed dataset 

# In[4]:


# Trim the merged dataset to include only years from 2000 to 2020
trimmed_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Topic']

# Add the year columns we want to keep (2000-2020)
for year in range(2000, 2021):
    year_col = f'year_{year}'
    if year_col in merged_df.columns:
        trimmed_cols.append(year_col)

# Create the trimmed dataset
trimmed_df = merged_df[trimmed_cols]
print(f"Trimmed dataframe shape: {trimmed_df.shape}")
print(f"Columns in trimmed dataframe: {trimmed_df.columns.tolist()}")

# Basic count of topics in the trimmed dataset
topic_counts = trimmed_df['Topic'].value_counts()
print(f"\nNumber of unique topics in 2000-2020 dataset: {len(topic_counts)}")
print("\nAll topics and their counts:")
print(topic_counts)

# Count missing topics (NA or null)
missing_topics_mask = trimmed_df['Topic'].isna() | (trimmed_df['Topic'] == "NA")
missing_count = missing_topics_mask.sum()
print(f"\nNumber of rows with missing topics (NA or null): {missing_count}")
print(f"Percentage of rows with missing topics: {missing_count/len(trimmed_df)*100:.2f}%")

# Tabulate indicator names for rows with missing topics
missing_indicators = trimmed_df.loc[missing_topics_mask, 'Indicator Name'].value_counts()
print(f"\nNumber of unique indicators with missing topics: {len(missing_indicators)}")
print("\nTop 20 indicators with missing topics:")
print(missing_indicators.head(20))
print("\nFull list of indicators with missing topics and their counts:")
print(missing_indicators)

# Plot indicators with missing topics
plt.figure(figsize=(14, 10))
missing_indicators.head(20).plot(kind='barh', color='crimson')
plt.title('Top 20 Indicators with Missing Topics (NA or null)')
plt.xlabel('Count')
plt.ylabel('Indicator Name')
plt.tight_layout()
plt.savefig('missing_topic_indicators_2000_2020.png')
plt.show()

# Save the full list to CSV for easier viewing
missing_indicators_df = pd.DataFrame({
    'Indicator Name': missing_indicators.index,
    'Count': missing_indicators.values
})
missing_indicators_df.to_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables/missing_topic_indicators_2000_2020.csv', index=False)
print("Full list of missing topic indicators saved to 'missing_topic_indicators_2000_2020.csv'")


# Replace missing topics with "Emissions that contribute to climate change"
trimmed_df.loc[missing_topics_mask, 'Topic'] = "Emissions that contribute to climate change"

# Verify the replacement
new_topic_count = (trimmed_df['Topic'] == "Emissions that contribute to climate change").sum()
print(f"\nNumber of rows with the new topic after replacement: {new_topic_count}")

# Check that there are no more missing topics
remaining_missing = trimmed_df['Topic'].isna().sum()
print(f"Remaining rows with null topics: {remaining_missing}")

# Updated topic counts
updated_topic_counts = trimmed_df['Topic'].value_counts()
print("\nUpdated topic counts after replacement:")
print(updated_topic_counts.head(10))

# Check where the new topic ranks
topic_rank = updated_topic_counts.index.get_loc("Emissions that contribute to climate change") + 1
print(f"\n'Emissions that contribute to climate change' is now rank #{topic_rank} by frequency")

# Display a sample of the updated rows
print("\nSample of rows with the updated topic:")
print(trimmed_df[trimmed_df['Topic'] == "Emissions that contribute to climate change"].head(5))


# In[6]:



# Save Topics to a CSV and look at them manually
current_topics = pd.DataFrame({
    'Topic': trimmed_df['Topic'].unique()
})
current_topics.to_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables/original_unique_topics.csv', index=False)
print(f"Saved {len(current_topics)} unique topics to 'original_unique_topics.csv'")

# Create a function to map topics to more coarse categories
def map_to_coarse_category(topic):
    if pd.isna(topic):
        return "Other"
    
    topic_lower = str(topic).lower()
    
    # Poverty & Inequality category
    if any(term in topic_lower for term in ['poverty', 'gini', 'inequality', 'income distribution', 
                                          'wealth distribution', 'income gap', 'wealth gap']):
        return "Poverty & Inequality"
    
    # Economic categories
    elif any(term in topic_lower for term in ['economic', 'economy', 'finance', 'trade', 'business']):
        return "Economy & Finance"
    
    elif any(term in topic_lower for term in ['market']):
        return "Market"
    
    # Environmental categories
    elif any(term in topic_lower for term in ['environment', 'climate', 'emission', 'pollution', 'forest', 'energy']):
        return "Environment & Climate"
    
    # Social categories (excluding poverty)
    elif any(term in topic_lower for term in ['health']):
        return "Health"
    
    elif any(term in topic_lower for term in ['education']):
        return "Education"
    
     # Social categories (excluding poverty)
    elif any(term in topic_lower for term in ['social', 'urban', 'rural']):
        return "Social Development"
    
    elif any(term in topic_lower for term in ['education']):
        return "Gender"
    
    # Infrastructure categories
    elif any(term in topic_lower for term in ['infrastructure', 'transport', 'communication', 'technology']):
        return "Infrastructure & Technology"
    
    # Government categories
    elif any(term in topic_lower for term in ['government', 'public sector', 'governance', 'law', 'regulation']):
        return "Government & Governance"
    
    # Agriculture categories
    elif any(term in topic_lower for term in ['agriculture', 'farming', 'land', 'food']):
        return "Agriculture & Food"
    
    # Default category
    else:
        return "Other"

# Create a new column with coarse categories
trimmed_df['Coarse_Topic'] = trimmed_df['Topic'].apply(map_to_coarse_category)

# Count the distribution of coarse topics
coarse_counts = trimmed_df['Coarse_Topic'].value_counts()
print("\nDistribution of coarse topic categories:")
print(coarse_counts)

# Visualize the coarse categories
plt.figure(figsize=(12, 8))
coarse_counts.plot(kind='bar', color='teal')
plt.title('Distribution of Coarse Topic Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('coarse_topic_distribution.png')
plt.show()

# Create a mapping table that shows which original topics were assigned to each coarse category
topic_mapping = trimmed_df.groupby('Coarse_Topic')['Topic'].unique().apply(list)
topic_mapping_df = pd.DataFrame({
    'Coarse_Topic': topic_mapping.index,
    'Original_Topics': [', '.join(topics[:5]) + ('...' if len(topics) > 5 else '') for topics in topic_mapping]
})
print("\nMapping of original topics to coarse categories (showing up to 5 original topics per category):")
print(topic_mapping_df)

# Save the complete mapping for manual review
complete_mapping = []
for coarse_topic, topics in zip(topic_mapping.index, topic_mapping):
    for original_topic in topics:
        complete_mapping.append({
            'Coarse_Topic': coarse_topic,
            'Original_Topic': original_topic,
            'Count': trimmed_df[trimmed_df['Topic'] == original_topic].shape[0]
        })
mapping_df = pd.DataFrame(complete_mapping).sort_values(['Coarse_Topic', 'Count'], ascending=[True, False])
mapping_df.to_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables/topic_category_mapping.csv', index=False)
print("\nComplete mapping of original topics to coarse categories saved to 'topic_category_mapping.csv'")

# Make sure to include Indicator name in the output
# Assuming trimmed_df has an 'Indicator_Name' column
# If not, you need to ensure this column is available in trimmed_df before this point

# Save the updated dataframe with coarse topics
trimmed_df.to_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables/trimmed_data_with_coarse_topics.csv', index=False)
print("Updated dataframe with coarse topics saved to 'trimmed_data_with_coarse_topics.csv'")


# # Missing Data

# In[7]:


# Check how many null values exist in each column
missing_counts = trimmed_df.isnull().sum()
print("Missing value counts by column:")
print(missing_counts)

# Calculate percentage of missing values
missing_percentage = (missing_counts / len(trimmed_df)) * 100
print("\nMissing value percentage by column:")
print(missing_percentage)

# Visualize missing data patterns
msno.matrix(trimmed_df)
plt.title('Missing Value Patterns')
plt.tight_layout()
plt.savefig('missing_data_pattern.png')
plt.show()

# Fill missing Topics with "Not Categorized"
trimmed_df['Topic'] = trimmed_df['Topic'].fillna("Not Categorized")

# Fill missing IndicatorName with "Not Categorized"
trimmed_df['Topic'] = trimmed_df['Indicator Name'].fillna("Not Categorized")

# The map_to_coarse_category function already handles NaN values and returns "Other"
# But you could explicitly set them if needed
trimmed_df['Coarse_Topic'] = trimmed_df['Coarse_Topic'].fillna("Other")


# In[10]:


from sklearn.impute import KNNImputer
from tqdm import tqdm  # for progress tracking (install with pip if needed)

# Sort your dataframe
trimmed_df = trimmed_df.sort_values(['Country Name', 'Country Code', 'Indicator Code', 'Indicator Name', 'Topic', 'Coarse_Topic'])

# Get year columns that need imputation
year_cols = [col for col in trimmed_df.columns if col.startswith('year_')]
print(f"Found {len(year_cols)} year columns for imputation")

# Make a copy for comparison
original_df = trimmed_df.copy()

# Count missing values before
missing_before = trimmed_df[year_cols].isna().sum().sum()
print(f"Missing values before imputation: {missing_before}")

# OPTIMIZATION APPROACH 1: Use fewer neighbors
imputer = KNNImputer(n_neighbors=3)  # Reduce from 5 to 3 neighbors

# OPTIMIZATION APPROACH 2: Impute by indicator type in smaller chunks
# Group by indicator code to process similar data together
groups = trimmed_df.groupby('Indicator Code')

# Create a new dataframe to store results
result_df = pd.DataFrame()

print("Performing KNN imputation by indicator groups...")
for name, group in tqdm(groups):
    # Only process groups with missing values
    if group[year_cols].isna().any().any():
        # Apply KNN imputation only to the year columns for this indicator
        year_data = group[year_cols].values
        # Only impute if we have enough data
        if len(year_data) > 3:  # Need at least more rows than neighbors
            try:
                imputed_data = imputer.fit_transform(year_data)
                group_copy = group.copy()
                group_copy[year_cols] = imputed_data
                result_df = pd.concat([result_df, group_copy])
            except Exception as e:
                # If KNN fails, use simpler method for this group
                print(f"KNN failed for {name}, using ffill/bfill instead: {e}")
                group_copy = group.copy()
                group_copy[year_cols] = group_copy[year_cols].fillna(method='ffill', axis=1)
                group_copy[year_cols] = group_copy[year_cols].fillna(method='bfill', axis=1)
                result_df = pd.concat([result_df, group_copy])
        else:
            # Too few samples, use simpler method
            group_copy = group.copy()
            group_copy[year_cols] = group_copy[year_cols].fillna(method='ffill', axis=1)
            group_copy[year_cols] = group_copy[year_cols].fillna(method='bfill', axis=1)
            result_df = pd.concat([result_df, group_copy])
    else:
        # No missing values, just add to result
        result_df = pd.concat([result_df, group])

# For any remaining NaN values, use forward/backward fill
if result_df[year_cols].isna().any().any():
    print("Applying ffill/bfill for any remaining missing values...")
    result_df[year_cols] = result_df[year_cols].fillna(method='ffill', axis=1)
    result_df[year_cols] = result_df[year_cols].fillna(method='bfill', axis=1)

# Count missing values after
missing_after = result_df[year_cols].isna().sum().sum()
print(f"Missing values after imputation: {missing_before - missing_after}")
print(f"Filled {missing_before - missing_after} values ({(missing_before - missing_after) / max(1, missing_before):.2%} of missing values)")

# Save the imputed dataset
result_df.to_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv', index=False)
print("Saved imputed dataset to 'trimmed_data_with_coarse_topics_knn_imputed.csv'")


# # Correlation Analysis

# In[4]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Get year columns
year_cols = [col for col in df.columns if col.startswith('year_')]
print(f"Found {len(year_cols)} year columns")

# Get unique coarse topics
coarse_topics = df['Coarse_Topic'].dropna().unique()
print(f"Found {len(coarse_topics)} unique coarse topics")

# Create output directory
output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables')
output_dir.mkdir(parents=True, exist_ok=True)

# Create dictionaries to store results
topic_avg_corr = {}  # Average absolute correlation per topic
topic_med_corr = {}  # Median absolute correlation per topic
topic_max_corr = {}  # Maximum absolute correlation per topic
topic_corr_count = {}  # Count of strong correlations (|r| >= 0.7)
topic_ind_count = {}  # Count of indicators per topic
topic_pair_count = {}  # Count of indicator pairs per topic
topic_strong_pairs = {}  # List of strong pairs per topic

# Process each coarse topic
for topic in coarse_topics:
    print(f"\nProcessing topic: {topic}")
    
    # Filter data for this coarse topic
    topic_df = df[df['Coarse_Topic'] == topic].copy()
    
    # Get unique indicators for this topic
    indicators = topic_df['Indicator Name'].unique()
    topic_ind_count[topic] = len(indicators)
    
    if len(indicators) < 2:
        print(f"Topic {topic} has fewer than 2 indicators, skipping")
        topic_avg_corr[topic] = 0
        topic_med_corr[topic] = 0
        topic_max_corr[topic] = 0
        topic_corr_count[topic] = 0
        topic_pair_count[topic] = 0
        topic_strong_pairs[topic] = []
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
        topic_avg_corr[topic] = 0
        topic_med_corr[topic] = 0
        topic_max_corr[topic] = 0
        topic_corr_count[topic] = 0
        topic_pair_count[topic] = 0
        topic_strong_pairs[topic] = []
        continue
    
    # Extract all pairwise correlations (excluding self-correlations)
    corr_values = []
    strong_pairs = []
    
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            ind1 = correlation.columns[i]
            ind2 = correlation.columns[j]
            corr_val = correlation.iloc[i, j]
            
            if pd.notna(corr_val):
                corr_values.append(abs(corr_val))  # Use absolute value
                
                # Track strong correlations
                if abs(corr_val) >= 0.7:
                    code1 = indicator_to_code.get(ind1, "Unknown")
                    code2 = indicator_to_code.get(ind2, "Unknown")
                    strong_pairs.append((ind1, code1, ind2, code2, corr_val))
    
    # Store results
    topic_pair_count[topic] = len(corr_values)
    
    if corr_values:
        topic_avg_corr[topic] = np.mean(corr_values)
        topic_med_corr[topic] = np.median(corr_values)
        topic_max_corr[topic] = np.max(corr_values)
        topic_corr_count[topic] = sum(1 for v in corr_values if v >= 0.7)
        topic_strong_pairs[topic] = sorted(strong_pairs, key=lambda x: abs(x[4]), reverse=True)
    else:
        topic_avg_corr[topic] = 0
        topic_med_corr[topic] = 0
        topic_max_corr[topic] = 0
        topic_corr_count[topic] = 0
        topic_strong_pairs[topic] = []

# Convert results to DataFrames
results_df = pd.DataFrame({
    'Coarse_Topic': list(topic_avg_corr.keys()),
    'Indicator_Count': [topic_ind_count[t] for t in topic_avg_corr.keys()],
    'Pair_Count': [topic_pair_count[t] for t in topic_avg_corr.keys()],
    'Avg_Abs_Correlation': [topic_avg_corr[t] for t in topic_avg_corr.keys()],
    'Median_Abs_Correlation': [topic_med_corr[t] for t in topic_avg_corr.keys()],
    'Max_Abs_Correlation': [topic_max_corr[t] for t in topic_avg_corr.keys()],
    'Strong_Correlation_Count': [topic_corr_count[t] for t in topic_avg_corr.keys()],
    'Strong_Correlation_Pct': [topic_corr_count[t]/max(1, topic_pair_count[t])*100 for t in topic_avg_corr.keys()]
})

# Sort by average absolute correlation (descending)
results_df = results_df.sort_values('Avg_Abs_Correlation', ascending=False)

# Save results to CSV
results_file = output_dir / "coarse_topic_internal_correlation.csv"
results_df.to_csv(results_file, index=False)
print(f"Saved coarse topic correlation metrics to {results_file}")

# Create visualization of average correlation by topic
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.barplot(x='Coarse_Topic', y='Avg_Abs_Correlation', data=results_df.head(15))
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.title('Average Internal Absolute Correlation by Coarse Topic (Top 15)')
plt.tight_layout()

plt.subplot(2, 1, 2)
sns.barplot(x='Coarse_Topic', y='Strong_Correlation_Pct', data=results_df.head(15))
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.title('Percentage of Strong Correlations (|r| ≥ 0.7) by Coarse Topic (Top 15)')
plt.tight_layout()

vis_file = output_dir / "coarse_topic_correlation_chart.png"
plt.savefig(vis_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {vis_file}")

# Create detailed report of strong correlations by topic
report_file = output_dir / "coarse_topic_strong_correlations.txt"
with open(report_file, 'w') as f:
    f.write("COARSE TOPIC INTERNAL CORRELATION ANALYSIS\n")
    f.write("=========================================\n\n")
    
    f.write("Topics Ranked by Average Internal Correlation\n")
    f.write("--------------------------------------------\n")
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        topic = row['Coarse_Topic']
        f.write(f"{i}. {topic}\n")
        f.write(f"   Indicators: {row['Indicator_Count']}\n")
        f.write(f"   Avg. Absolute Correlation: {row['Avg_Abs_Correlation']:.4f}\n")
        f.write(f"   Median Absolute Correlation: {row['Median_Abs_Correlation']:.4f}\n")
        f.write(f"   Max Absolute Correlation: {row['Max_Abs_Correlation']:.4f}\n")
        f.write(f"   Strong Correlations: {row['Strong_Correlation_Count']} of {row['Pair_Count']} pairs ({row['Strong_Correlation_Pct']:.1f}%)\n\n")
    
    f.write("\n\nDETAILED STRONG CORRELATIONS BY TOPIC\n")
    f.write("====================================\n\n")
    
    # Process each topic ordered by avg correlation
    for _, row in results_df.iterrows():
        topic = row['Coarse_Topic']
        strong_pairs = topic_strong_pairs[topic]
        
        if strong_pairs:
            f.write(f"\n\nTOPIC: {topic}\n")
            f.write("=" * (len(topic) + 7) + "\n")
            f.write(f"Average Absolute Correlation: {row['Avg_Abs_Correlation']:.4f}\n")
            f.write(f"Strong Correlations: {len(strong_pairs)} of {row['Pair_Count']} pairs ({row['Strong_Correlation_Pct']:.1f}%)\n\n")
            
            f.write("Top Strongly Correlated Indicator Pairs:\n")
            f.write("-" * 40 + "\n")
            
            # Display top 10 strongest correlations (or all if fewer)
            for pair_num, (ind1, code1, ind2, code2, corr_val) in enumerate(strong_pairs[:10], 1):
                direction = "+" if corr_val > 0 else "-"
                f.write(f"{pair_num}. {ind1} [{code1}] and\n   {ind2} [{code2}]\n   {direction}r = {abs(corr_val):.4f}\n\n")

print(f"Saved detailed report to {report_file}")

# Create correlation strength heatmap
# Only include topics with at least 5 indicators
topics_for_heatmap = results_df[results_df['Indicator_Count'] >= 5].head(15)['Coarse_Topic'].tolist()

if topics_for_heatmap:
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    for topic in topics_for_heatmap:
        row = results_df[results_df['Coarse_Topic'] == topic].iloc[0]
        heatmap_data.append({
            'Coarse_Topic': topic,
            'Avg_Correlation': row['Avg_Abs_Correlation'],
            'Pct_Strong': row['Strong_Correlation_Pct'] / 100,  # Convert to decimal
            'Indicator_Count': row['Indicator_Count']
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    pivot_table = heatmap_df.pivot_table(
        index='Coarse_Topic', 
        values=['Avg_Correlation', 'Pct_Strong', 'Indicator_Count'],
        aggfunc='first'
    ).sort_values('Avg_Correlation', ascending=False)
    
    # Plot heatmap for average correlation
    plt.subplot(1, 2, 1)
    sns.heatmap(
        pivot_table[['Avg_Correlation']], 
        annot=True, 
        cmap='YlOrRd', 
        fmt='.3f',
        cbar_kws={'label': 'Average |r|'}
    )
    plt.title('Average Absolute Correlation\nby Coarse Topic')
    
    # Plot heatmap for percentage of strong correlations
    plt.subplot(1, 2, 2)
    sns.heatmap(
        pivot_table[['Pct_Strong']], 
        annot=True, 
        cmap='YlOrRd', 
        fmt='.1%',
        cbar_kws={'label': '% Strong Correlations'}
    )
    plt.title('Percentage of Strong Correlations\n(|r| ≥ 0.7) by Coarse Topic')
    
    plt.tight_layout()
    
    heatmap_file = output_dir / "coarse_topic_correlation_heatmap.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Saved correlation heatmap to {heatmap_file}")

print("\nAnalysis complete!")


# In[14]:


import pandas as pd
from pathlib import Path

# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Create output directory
output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Extract indicators in the Poverty & Inequality coarse topic
poverty_df = df[df['Coarse_Topic'] == 'Poverty & Inequality'].copy()
poverty_indicators = poverty_df['Indicator Name'].unique()

print(f"Found {len(poverty_indicators)} indicators in Poverty & Inequality coarse topic:")
for i, indicator in enumerate(poverty_indicators, 1):
    indicator_code = poverty_df[poverty_df['Indicator Name'] == indicator]['Indicator Code'].iloc[0]
    print(f"{i}. {indicator} [{indicator_code}]")

# Find Gini Index indicator
gini_index = None
for indicator in poverty_indicators:
    if 'gini' in indicator.lower():
        gini_index = indicator
        break

if not gini_index:
    print("Warning: Could not find Gini Index indicator")
    # If there are multiple potential matches, you could manually specify here
else:
    print(f"\nIdentified Gini Index: {gini_index}")
    
    # Create filtered dataframe with only the Gini Index from Poverty & Inequality
    gini_df = poverty_df[poverty_df['Indicator Name'] == gini_index].copy()
    
    # Create a final dataframe combining Gini with all non-Poverty indicators
    non_poverty_df = df[df['Coarse_Topic'] != 'Poverty & Inequality'].copy()
    final_df = pd.concat([gini_df, non_poverty_df])
    
    # Save filtered dataset
    filtered_file = output_dir / "dataset_with_only_gini_from_poverty.csv"
    final_df.to_csv(filtered_file, index=False)
    print(f"Saved filtered dataset to {filtered_file}")


# In[ ]:


import pandas as pd
from pathlib import Path

# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Create output directory
output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Extract indicators in the Poverty & Inequality coarse topic
poverty_df = df[df['Coarse_Topic'] == 'Poverty & Inequality'].copy()
poverty_indicators = poverty_df['Indicator Name'].unique()

print(f"Found {len(poverty_indicators)} indicators in Poverty & Inequality coarse topic:")
for i, indicator in enumerate(poverty_indicators, 1):
    indicator_code = poverty_df[poverty_df['Indicator Name'] == indicator]['Indicator Code'].iloc[0]
    print(f"{i}. {indicator} [{indicator_code}]")

# Find Gini Index indicator
gini_index = None
for indicator in poverty_indicators:
    if 'gini' in indicator.lower():
        gini_index = indicator
        break

if not gini_index:
    print("Warning: Could not find Gini Index indicator")
    # If there are multiple potential matches, you could manually specify here
else:
    print(f"\nIdentified Gini Index: {gini_index}")
    
    # Create filtered dataframe with only the Gini Index from Poverty & Inequality
    gini_df = poverty_df[poverty_df['Indicator Name'] == gini_index].copy()
    
    # Create a final dataframe combining Gini with all non-Poverty indicators
    non_poverty_df = df[df['Coarse_Topic'] != 'Poverty & Inequality'].copy()
    final_df = pd.concat([gini_df, non_poverty_df])
    
    # Save filtered dataset
    filtered_file = output_dir / "dataset_with_only_gini_from_poverty.csv"
    final_df.to_csv(filtered_file, index=False)
    print(f"Saved filtered dataset to {filtered_file}")


# In[18]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#gini-coefficient and other outcome variables.
# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_tables/trimmed_data_with_coarse_topics.csv'
# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_table/')
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
filtered_file = output_dir / "gini_filtered_dataset_without_correlated_indicators.csv"
filtered_df.to_csv(filtered_file, index=False)
print(f"Saved filtered dataset to {filtered_file}")

print("\nAnalysis complete!")


# For each coarse topic, calculate pairwise correlations between all indicators and identifies highly correlated pairs (using a threshold of |r| ≥ 0.7). The algorithm for removing highly correlated indicators works as follows:
# 
# Calculate the absolute correlation between all indicator pairs
# Identify pairs with correlation ≥ 0.7 (configurable threshold)
# Count how many times each indicator appears in high-correlation pairs
# Iteratively remove indicators with the highest frequency in high-correlation pairs
# Stop when no highly correlated pairs remain
# 
# This approach minimizes the number of indicators removed while ensuring that no remaining indicators are highly correlated.

# In[19]:


#Remove poverty variables other than outcome variable [bottom 10% share] and high-correlated variables.
import pandas as pd
from pathlib import Path

# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Create output directory
output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Extract indicators in the Poverty & Inequality coarse topic
poverty_df = df[df['Coarse_Topic'] == 'Poverty & Inequality'].copy()
poverty_indicators = poverty_df['Indicator Name'].unique()
print(f"Found {len(poverty_indicators)} indicators in Poverty & Inequality coarse topic:")
for i, indicator in enumerate(poverty_indicators, 1):
    indicator_code = poverty_df[poverty_df['Indicator Name'] == indicator]['Indicator Code'].iloc[0]
    print(f"{i}. {indicator} [{indicator_code}]")

# Instead of Gini, find "Income share held by highest 10%"
target_indicator = "Income share held by highest 10%"

# Check if the target indicator exists
if target_indicator in poverty_indicators:
    print(f"\nFound target indicator: {target_indicator}")
    
    # Create filtered dataframe with only the target indicator from Poverty & Inequality
    target_df = poverty_df[poverty_df['Indicator Name'] == target_indicator].copy()
    
    # Create a final dataframe combining target indicator with all non-Poverty indicators
    non_poverty_df = df[df['Coarse_Topic'] != 'Poverty & Inequality'].copy()
    final_df = pd.concat([target_df, non_poverty_df])
    
    # Save filtered dataset
    filtered_file = output_dir / "dataset_with_only_top10pct_from_poverty.csv"
    final_df.to_csv(filtered_file, index=False)
    print(f"Saved filtered dataset to {filtered_file}")
else:
    print(f"Warning: Could not find '{target_indicator}' in poverty indicators")
    
    # List similar indicators that might match what we're looking for
    similar_indicators = [ind for ind in poverty_indicators if 'income' in ind.lower() and ('10' in ind or 'ten' in ind.lower())]
    if similar_indicators:
        print("Found similar indicators that might match:")
        for ind in similar_indicators:
            print(f"- {ind}")


# In[21]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#Income share held by highest 10% and other outcome variables.
# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/dataset_with_only_top10pct_from_poverty.csv'
# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_table/')
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
filtered_file = output_dir / "topten_filtered_dataset_without_correlated_indicators.csv"
filtered_df.to_csv(filtered_file, index=False)
print(f"Saved filtered dataset to {filtered_file}")

print("\nAnalysis complete!")


# In[22]:


#Remove poverty variables other than outcome variable [Gini Coefficient] and high-correlated variables.
import pandas as pd
from pathlib import Path

# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/trimmed_data_with_coarse_topics_knn_imputed.csv'

# Create output directory
output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Extract indicators in the Poverty & Inequality coarse topic
poverty_df = df[df['Coarse_Topic'] == 'Poverty & Inequality'].copy()
poverty_indicators = poverty_df['Indicator Name'].unique()
print(f"Found {len(poverty_indicators)} indicators in Poverty & Inequality coarse topic:")
for i, indicator in enumerate(poverty_indicators, 1):
    indicator_code = poverty_df[poverty_df['Indicator Name'] == indicator]['Indicator Code'].iloc[0]
    print(f"{i}. {indicator} [{indicator_code}]")

# Instead of Gini, find "Income share held by highest 10%"
target_indicator = "Income share held by lowest 10%"

# Check if the target indicator exists
if target_indicator in poverty_indicators:
    print(f"\nFound target indicator: {target_indicator}")
    
    # Create filtered dataframe with only the target indicator from Poverty & Inequality
    target_df = poverty_df[poverty_df['Indicator Name'] == target_indicator].copy()
    
    # Create a final dataframe combining target indicator with all non-Poverty indicators
    non_poverty_df = df[df['Coarse_Topic'] != 'Poverty & Inequality'].copy()
    final_df = pd.concat([target_df, non_poverty_df])
    
    # Save filtered dataset
    filtered_file = output_dir / "dataset_with_only_bottom10pct_from_poverty.csv"
    final_df.to_csv(filtered_file, index=False)
    print(f"Saved filtered dataset to {filtered_file}")
else:
    print(f"Warning: Could not find '{target_indicator}' in poverty indicators")
    
    # List similar indicators that might match what we're looking for
    similar_indicators = [ind for ind in poverty_indicators if 'income' in ind.lower() and ('10' in ind or 'ten' in ind.lower())]
    if similar_indicators:
        print("Found similar indicators that might match:")
        for ind in similar_indicators:
            print(f"- {ind}")


# In[23]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#Income share held by highest 10% and other outcome variables.
# Define file path - replace with your actual file path
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/1_data/dataset_with_only_bottom10pct_from_poverty.csv'
# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

output_dir = Path('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni_analysis/3_table/')
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
filtered_file = output_dir / "bottomten_filtered_dataset_without_correlated_indicators.csv"
filtered_df.to_csv(filtered_file, index=False)
print(f"Saved filtered dataset to {filtered_file}")

print("\nAnalysis complete!")


# In[ ]:




