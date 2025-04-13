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


# # How far back should we go with the indicators? 

# In[9]:


# Load the data
df = pd.read_csv('/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni Analysis/0_data/WDI_CSV_2025_01_28/WDICSV.csv')
print("Original column names:", df.columns.tolist())

# Calculate data availability for each year column
years = []
data_counts = []

# Process each year column (1960-2023)
for year in range(1960, 2024):
    year_str = str(year)
    if year_str in df.columns:
        # Count non-null values
        count = df[year_str].notna().sum()
        years.append(year)
        data_counts.append(count)

# Create a basic plot
plt.figure(figsize=(12, 6))
plt.bar(years, data_counts, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Available Data Points')
plt.title('Data Availability Over Time')
plt.xticks(np.arange(1960, 2024, 5), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Shade different time periods
plt.axvspan(2000, 2020, alpha=0.2, color='green', label='High Coverage (2000-2020)')
plt.axvspan(1990, 2000, alpha=0.2, color='yellow', label='Good Coverage (1990-2000)')
plt.axvspan(1970, 1990, alpha=0.2, color='orange', label='Moderate Coverage (1970-1990)')
plt.legend()

plt.tight_layout()
plt.savefig('data_availability.png')
plt.show()

# Group by decade for summary
decade_data = {}
for i, year in enumerate(years):
    decade = (year // 10) * 10
    if decade not in decade_data:
        decade_data[decade] = []
    decade_data[decade].append(data_counts[i])

# Print decade summary
print("\nData Availability by Decade:")
for decade, counts in sorted(decade_data.items()):
    avg_count = sum(counts) / len(counts)
    min_count = min(counts)
    max_count = max(counts)
    print(f"{decade}s: Avg={avg_count:.1f}, Min={min_count}, Max={max_count}")

# Find best decade
best_decade = max(decade_data.items(), key=lambda x: sum(x[1])/len(x[1]))[0]

print("\nRecommendation:")
print(f"Based on data availability, focus on data from {best_decade} onwards.")
print("The period 2000-2020 offers the most comprehensive coverage.")


# # Exploratory analysis on the period 2000-2020. 

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File path - update this to your file location
file_path = '/Users/varnithakurli/Documents/GitHub/wdi-income-inequality/Varni Analysis/0_data/WDI_CSV_2025_01_28/WDICSV.csv'

# Create output directory if it doesn't exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Load the dataset
print("Loading WDI dataset...")
df = pd.read_csv(file_path)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
print("Original column names:", df.columns.tolist())

# Indicators for income inequality
TOP_10_CODE = 'SI.DST.10TH.10'      # Income share held by highest 10%
BOTTOM_10_CODE = 'SI.DST.FRST.10'   # Income share held by lowest 10%
GINI_CODE = 'SI.POV.GINI'           # Gini coefficient

# Define column names based on the provided information
metadata_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']

# Extract data for 2000-2020 period (better data coverage)
year_cols = [str(year) for year in range(2000, 2021)]
selected_cols = metadata_cols + year_cols

# Filter for the indicators we want
mask = df['Indicator Code'].isin([TOP_10_CODE, BOTTOM_10_CODE, GINI_CODE])
inequality_df = df.loc[mask].copy()
print(f"Filtered data for inequality indicators: {inequality_df.shape[0]} rows")

# Reshape from wide to long format for easier analysis
if inequality_df.shape[0] > 0:
    # Select only the columns we need
    available_cols = [col for col in selected_cols if col in inequality_df.columns]
    # Check which year columns actually have data
    year_cols_with_data = [col for col in year_cols if col in available_cols]
    
    if year_cols_with_data:
        long_df = pd.melt(
            inequality_df,
            id_vars=[col for col in metadata_cols if col in available_cols],
            value_vars=year_cols_with_data,
            var_name='Year',
            value_name='Value'
        )
        long_df['Year'] = long_df['Year'].astype(int)
        print(f"Reshaped to long format: {long_df.shape[0]} rows")
    else:
        print("Warning: No year columns found with data")
        long_df = pd.DataFrame(columns=metadata_cols + ['Year', 'Value'])
else:
    print("Warning: No data found for the specified indicators")
    long_df = pd.DataFrame(columns=metadata_cols + ['Year', 'Value'])

# Separate data for each indicator
if long_df.shape[0] > 0:
    top10_df = long_df[long_df['Indicator Code'] == TOP_10_CODE].copy()
    bottom10_df = long_df[long_df['Indicator Code'] == BOTTOM_10_CODE].copy()
    gini_df = long_df[long_df['Indicator Code'] == GINI_CODE].copy()
    
    # Print summary of data availability
    print(f"Top 10% income share data points: {top10_df['Value'].count()}")
    print(f"Bottom 10% income share data points: {bottom10_df['Value'].count()}")
    print(f"Gini coefficient data points: {gini_df['Value'].count()}")
else:
    top10_df = pd.DataFrame(columns=metadata_cols + ['Year', 'Value'])
    bottom10_df = pd.DataFrame(columns=metadata_cols + ['Year', 'Value'])
    gini_df = pd.DataFrame(columns=metadata_cols + ['Year', 'Value'])
    print("No data points found for any indicator")

# Count data points per country for each indicator
if not top10_df.empty:
    top10_counts = top10_df.groupby('Country Name')['Value'].count().sort_values(ascending=False)
    print("\nTop 10 countries with most data points for Top 10% income share:")
    print(top10_counts.head(10))
else:
    top10_counts = pd.Series(dtype=float)
    print("\nNo data found for Top 10% income share")

if not bottom10_df.empty:
    bottom10_counts = bottom10_df.groupby('Country Name')['Value'].count().sort_values(ascending=False)
    print("\nTop 10 countries with most data points for Bottom 10% income share:")
    print(bottom10_counts.head(10))
else:
    bottom10_counts = pd.Series(dtype=float)
    print("\nNo data found for Bottom 10% income share")

if not gini_df.empty:
    gini_counts = gini_df.groupby('Country Name')['Value'].count().sort_values(ascending=False)
    print("\nTop 10 countries with most data points for Gini coefficient:")
    print(gini_counts.head(10))
else:
    gini_counts = pd.Series(dtype=float)
    print("\nNo data found for Gini coefficient")

# Find countries that have data for all three indicators
top10_countries = set(top10_counts.index) if not top10_counts.empty else set()
bottom10_countries = set(bottom10_counts.index) if not bottom10_counts.empty else set()
gini_countries = set(gini_counts.index) if not gini_counts.empty else set()

countries_with_all = top10_countries & bottom10_countries & gini_countries
print(f"\nFound {len(countries_with_all)} countries with data for all three indicators")

# Define a minimum threshold for data completeness (e.g., at least 5 years of data)
MIN_DATA_POINTS = 5

# Countries with sufficient data for each indicator
countries_with_enough_top10 = {country for country in top10_countries if top10_counts[country] >= MIN_DATA_POINTS}
countries_with_enough_bottom10 = {country for country in bottom10_countries if bottom10_counts[country] >= MIN_DATA_POINTS}
countries_with_enough_gini = {country for country in gini_countries if gini_counts[country] >= MIN_DATA_POINTS}

print(f"\nFound {len(countries_with_enough_top10)} countries with at least {MIN_DATA_POINTS} years of top 10% data")
print(f"Found {len(countries_with_enough_bottom10)} countries with at least {MIN_DATA_POINTS} years of bottom 10% data")
print(f"Found {len(countries_with_enough_gini)} countries with at least {MIN_DATA_POINTS} years of Gini data")

# Countries with sufficient data for all indicators
countries_with_enough_all = countries_with_enough_top10 & countries_with_enough_bottom10 & countries_with_enough_gini
print(f"Found {len(countries_with_enough_all)} countries with at least {MIN_DATA_POINTS} years of data for all indicators")

#
# TOP 10% INCOME SHARE ANALYSIS
#
print("\n---- TOP 10% INCOME SHARE ANALYSIS ----")

if top10_df.empty or len(countries_with_enough_top10) == 0:
    print("Insufficient data for Top 10% income share analysis")
else:
    # Find countries with highest and lowest top 10% income share
    country_top10_means = {}

    for country in countries_with_enough_top10:
        # Get data for this country
        country_data = top10_df[top10_df['Country Name'] == country]
        
        # Calculate mean income share
        mean_income_share = country_data['Value'].mean()
        
        # Only keep if we have data
        if not np.isnan(mean_income_share):
            country_top10_means[country] = mean_income_share

    # Sort countries by top 10% income share
    sorted_top10 = sorted(country_top10_means.items(), key=lambda x: x[1], reverse=True)

    # Get countries with highest and lowest top 10% income share
    highest_top10 = sorted_top10[:10]
    lowest_top10 = sorted_top10[-10:]

    print("\nCountries with highest income share held by top 10%:")
    for i, (country, share) in enumerate(highest_top10, 1):
        print(f"{i}. {country}: {share:.1f}%")

    print("\nCountries with lowest income share held by top 10%:")
    for i, (country, share) in enumerate(lowest_top10, 1):
        print(f"{i}. {country}: {share:.1f}%")

    # Plot top 10% income share trends for top 5 countries
    if len(highest_top10) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in highest_top10[:5]:
            country_data = top10_df[top10_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Top 10% Income Share Trends for Most Unequal Countries")
        plt.xlabel("Year")
        plt.ylabel("Income Share (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "top10_trends_highest.png")
        print("Saved plot: top10_trends_highest.png")
    
    # Plot top 10% income share trends for bottom 5 countries
    if len(lowest_top10) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in lowest_top10[:5]:
            country_data = top10_df[top10_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Top 10% Income Share Trends for Most Equal Countries")
        plt.xlabel("Year")
        plt.ylabel("Income Share (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "top10_trends_lowest.png")
        print("Saved plot: top10_trends_lowest.png")

#
# BOTTOM 10% INCOME SHARE ANALYSIS
#
print("\n---- BOTTOM 10% INCOME SHARE ANALYSIS ----")

if bottom10_df.empty or len(countries_with_enough_bottom10) == 0:
    print("Insufficient data for Bottom 10% income share analysis")
else:
    # Find countries with highest and lowest bottom 10% income share
    country_bottom10_means = {}

    for country in countries_with_enough_bottom10:
        # Get data for this country
        country_data = bottom10_df[bottom10_df['Country Name'] == country]
        
        # Calculate mean income share
        mean_income_share = country_data['Value'].mean()
        
        # Only keep if we have data
        if not np.isnan(mean_income_share):
            country_bottom10_means[country] = mean_income_share

    # Sort countries by bottom 10% income share (higher is better for equality)
    sorted_bottom10 = sorted(country_bottom10_means.items(), key=lambda x: x[1], reverse=True)

    # Get countries with highest and lowest bottom 10% income share
    highest_bottom10 = sorted_bottom10[:10]
    lowest_bottom10 = sorted_bottom10[-10:]

    print("\nCountries with highest income share held by bottom 10% (more equal):")
    for i, (country, share) in enumerate(highest_bottom10, 1):
        print(f"{i}. {country}: {share:.1f}%")

    print("\nCountries with lowest income share held by bottom 10% (less equal):")
    for i, (country, share) in enumerate(lowest_bottom10, 1):
        print(f"{i}. {country}: {share:.1f}%")

    # Plot bottom 10% income share trends for top 5 countries
    if len(highest_bottom10) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in highest_bottom10[:5]:
            country_data = bottom10_df[bottom10_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Bottom 10% Income Share Trends for Most Equal Countries")
        plt.xlabel("Year")
        plt.ylabel("Income Share (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "bottom10_trends_highest.png")
        print("Saved plot: bottom10_trends_highest.png")
    
    # Plot bottom 10% income share trends for bottom 5 countries
    if len(lowest_bottom10) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in lowest_bottom10[:5]:
            country_data = bottom10_df[bottom10_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Bottom 10% Income Share Trends for Most Unequal Countries")
        plt.xlabel("Year")
        plt.ylabel("Income Share (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "bottom10_trends_lowest.png")
        print("Saved plot: bottom10_trends_lowest.png")

#
# GINI COEFFICIENT ANALYSIS
#
print("\n---- GINI COEFFICIENT ANALYSIS ----")

if gini_df.empty or len(countries_with_enough_gini) == 0:
    print("Insufficient data for Gini coefficient analysis")
else:
    # Find countries with highest and lowest Gini coefficients
    country_gini_means = {}

    for country in countries_with_enough_gini:
        # Get data for this country
        country_data = gini_df[gini_df['Country Name'] == country]
        
        # Calculate mean Gini coefficient
        mean_gini = country_data['Value'].mean()
        
        # Only keep if we have data
        if not np.isnan(mean_gini):
            country_gini_means[country] = mean_gini

    # Sort countries by Gini coefficient (lower is better for equality)
    sorted_gini = sorted(country_gini_means.items(), key=lambda x: x[1])

    # Get countries with lowest and highest Gini coefficients
    lowest_gini = sorted_gini[:10]
    highest_gini = sorted_gini[-10:]

    print("\nCountries with lowest Gini coefficients (most equal):")
    for i, (country, gini) in enumerate(lowest_gini, 1):
        print(f"{i}. {country}: {gini:.1f}")

    print("\nCountries with highest Gini coefficients (most unequal):")
    for i, (country, gini) in enumerate(highest_gini, 1):
        print(f"{i}. {country}: {gini:.1f}")

    # Plot Gini coefficient trends for most equal countries
    if len(lowest_gini) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in lowest_gini[:5]:
            country_data = gini_df[gini_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Gini Coefficient Trends for Most Equal Countries")
        plt.xlabel("Year")
        plt.ylabel("Gini Coefficient")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "gini_trends_lowest.png")
        print("Saved plot: gini_trends_lowest.png")
    
    # Plot Gini coefficient trends for most unequal countries
    if len(highest_gini) >= 5:
        plt.figure(figsize=(12, 6))
        for country, _ in highest_gini[:5]:
            country_data = gini_df[gini_df['Country Name'] == country]
            plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', label=country)

        plt.title("Gini Coefficient Trends for Most Unequal Countries")
        plt.xlabel("Year")
        plt.ylabel("Gini Coefficient")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "gini_trends_highest.png")
        print("Saved plot: gini_trends_highest.png")

# Time series analysis
print("\n---- TIME SERIES ANALYSIS ----")

# Calculate global averages for each indicator over time
if not top10_df.empty:
    global_top10 = top10_df.groupby('Year')['Value'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(global_top10.index.to_numpy(), global_top10.values.to_numpy(), marker='o', label='Top 10% Income Share')
    plt.title("Global Trend in Top 10% Income Share")
    plt.xlabel("Year")
    plt.ylabel("Income Share (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "global_top10_trend.png")
    print("Saved plot: global_top10_trend.png")

if not bottom10_df.empty:
    global_bottom10 = bottom10_df.groupby('Year')['Value'].mean()
    plt.figure(figsize=(12, 6))
    # For bottom 10% global trend
    plt.plot(global_bottom10.index.to_numpy(), global_bottom10.values.to_numpy(), marker='o', label='Bottom 10% Income Share')
    plt.title("Global Trend in Bottom 10% Income Share")
    plt.xlabel("Year")
    plt.ylabel("Income Share (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "global_bottom10_trend.png")
    print("Saved plot: global_bottom10_trend.png")

if not gini_df.empty:
    global_gini = gini_df.groupby('Year')['Value'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(global_gini.index.to_numpy(), global_gini.values.to_numpy(), marker='o', label='Gini Coefficient')
    plt.title("Global Trend in Gini Coefficient")
    plt.xlabel("Year")
    plt.ylabel("Gini Coefficient")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "global_gini_trend.png")
    print("Saved plot: global_gini_trend.png")

print("\nAnalysis complete. Results saved to output directory.")


# # Correlations between the three Outcome variables

# In[37]:


# Add this section near the end of your script, just before the "Analysis complete" message

print("\n---- CORRELATION ANALYSIS ----")

# Check if we have enough data for correlation analysis
if countries_with_all and any(len(countries_with_enough_all) > 0 for _ in [1]):
    print(f"Performing correlation analysis using {len(countries_with_enough_all)} countries with sufficient data")
    
    # Create a dataframe to store the average values for each country and indicator
    correlation_data = []
    
    for country in countries_with_enough_all:
        # Get mean values for each indicator for this country
        top10_mean = top10_df[top10_df['Country Name'] == country]['Value'].mean()
        bottom10_mean = bottom10_df[bottom10_df['Country Name'] == country]['Value'].mean()
        gini_mean = gini_df[gini_df['Country Name'] == country]['Value'].mean()
        
        correlation_data.append({
            'Country': country,
            'Top 10%': top10_mean,
            'Bottom 10%': bottom10_mean,
            'Gini': gini_mean
        })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_data)
    corr_df.set_index('Country', inplace=True)
    
    # Calculate correlation matrix
    correlation_matrix = corr_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Create scatter plots for each pair of indicators
    plt.figure(figsize=(18, 6))
    
    # Top 10% vs Bottom 10%
    plt.subplot(1, 3, 1)
    plt.scatter(corr_df['Top 10%'].to_numpy(), corr_df['Bottom 10%'].to_numpy(), alpha=0.7)
    
    # Add regression line
    if len(corr_df) > 1:  # Need at least 2 points for regression
        z = np.polyfit(corr_df['Top 10%'].to_numpy(), corr_df['Bottom 10%'].to_numpy(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(corr_df['Top 10%'].min(), corr_df['Top 10%'].max(), 100)
        plt.plot(x_range, p(x_range), "r--")
    
    plt.title(f"Top 10% vs Bottom 10% (r = {correlation_matrix.loc['Top 10%', 'Bottom 10%']:.2f})")
    plt.xlabel("Top 10% Income Share (%)")
    plt.ylabel("Bottom 10% Income Share (%)")
    plt.grid(True, alpha=0.3)
    
    # Top 10% vs Gini
    plt.subplot(1, 3, 2)
    plt.scatter(corr_df['Top 10%'].to_numpy(), corr_df['Gini'].to_numpy(), alpha=0.7)
    
    # Add regression line
    if len(corr_df) > 1:
        z = np.polyfit(corr_df['Top 10%'].to_numpy(), corr_df['Gini'].to_numpy(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(corr_df['Top 10%'].min(), corr_df['Top 10%'].max(), 100)
        plt.plot(x_range, p(x_range), "r--")
    
    plt.title(f"Top 10% vs Gini (r = {correlation_matrix.loc['Top 10%', 'Gini']:.2f})")
    plt.xlabel("Top 10% Income Share (%)")
    plt.ylabel("Gini Coefficient")
    plt.grid(True, alpha=0.3)
    
    # Bottom 10% vs Gini
    plt.subplot(1, 3, 3)
    plt.scatter(corr_df['Bottom 10%'].to_numpy(), corr_df['Gini'].to_numpy(), alpha=0.7)
    
    # Add regression line
    if len(corr_df) > 1:
        z = np.polyfit(corr_df['Bottom 10%'].to_numpy(), corr_df['Gini'].to_numpy(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(corr_df['Bottom 10%'].min(), corr_df['Bottom 10%'].max(), 100)
        plt.plot(x_range, p(x_range), "r--")
    
    plt.title(f"Bottom 10% vs Gini (r = {correlation_matrix.loc['Bottom 10%', 'Gini']:.2f})")
    plt.xlabel("Bottom 10% Income Share (%)")
    plt.ylabel("Gini Coefficient")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "indicator_correlations.png")
    print("Saved plot: indicator_correlations.png")
    
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = plt.cm.RdBu_r
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    print("Saved plot: correlation_heatmap.png")
    
    # Add time-series correlation analysis (if data available for multiple years)
    print("\nAnalyzing correlations over time...")
    
    # For each year, calculate correlations between indicators across countries
    yearly_correlations = {}
    
    # Get common years across all indicators
    all_years = set(top10_df['Year'].unique()) & set(bottom10_df['Year'].unique()) & set(gini_df['Year'].unique())
    all_years = sorted(list(all_years))
    
    if len(all_years) > 1:  # Need at least 2 years for trend analysis
        # For each year, calculate correlations
        for year in all_years:
            # Get data for this year
            top10_year = top10_df[top10_df['Year'] == year]
            bottom10_year = bottom10_df[bottom10_df['Year'] == year]
            gini_year = gini_df[gini_df['Year'] == year]
            
            # Find common countries with data in this year
            top10_countries_year = set(top10_year['Country Name'])
            bottom10_countries_year = set(bottom10_year['Country Name'])
            gini_countries_year = set(gini_year['Country Name'])
            
            common_countries_year = top10_countries_year & bottom10_countries_year & gini_countries_year
            
            # Check if we have enough countries for correlation
            if len(common_countries_year) >= 5:  # Arbitrary threshold
                # Create yearly dataframe
                year_data = []
                
                for country in common_countries_year:
                    top10_val = float(top10_year[top10_year['Country Name'] == country]['Value'].iloc[0])
                    bottom10_val = float(bottom10_year[bottom10_year['Country Name'] == country]['Value'].iloc[0])
                    gini_val = float(gini_year[gini_year['Country Name'] == country]['Value'].iloc[0])
                    
                    year_data.append({
                        'Country': country,
                        'Top 10%': top10_val,
                        'Bottom 10%': bottom10_val,
                        'Gini': gini_val
                    })
                
                # Convert to DataFrame and calculate correlations
                year_df = pd.DataFrame(year_data)
                year_df.set_index('Country', inplace=True)
                
                year_corr = year_df.corr()
                
                # Store correlations
                yearly_correlations[year] = {
                    'Top10_Bottom10': year_corr.loc['Top 10%', 'Bottom 10%'],
                    'Top10_Gini': year_corr.loc['Top 10%', 'Gini'],
                    'Bottom10_Gini': year_corr.loc['Bottom 10%', 'Gini'],
                    'CountriesCount': len(common_countries_year)
                }
        
        # Create dataframe for yearly correlations
        if yearly_correlations:
            yearly_corr_df = pd.DataFrame.from_dict(yearly_correlations, orient='index')
            print(f"Generated correlation time series for {len(yearly_correlations)} years")
            
            # Plot correlation trends over time
            plt.figure(figsize=(12, 8))
            plt.plot(yearly_corr_df.index.to_numpy(), yearly_corr_df['Top10_Bottom10'].to_numpy(), 'o-', label='Top 10% vs Bottom 10%')
            plt.plot(yearly_corr_df.index.to_numpy(), yearly_corr_df['Top10_Gini'].to_numpy(), 's-', label='Top 10% vs Gini')
            plt.plot(yearly_corr_df.index.to_numpy(), yearly_corr_df['Bottom10_Gini'].to_numpy(), '^-', label='Bottom 10% vs Gini')
            
            plt.title("Correlation Trends Over Time")
            plt.xlabel("Year")
            plt.ylabel("Correlation Coefficient")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.savefig(output_dir / "correlation_trends.png")
            print("Saved plot: correlation_trends.png")
            
            # Plot number of countries with data over time
            plt.figure(figsize=(10, 6))
            plt.plot(yearly_corr_df.index.to_numpy(), yearly_corr_df['CountriesCount'].to_numpy(), 'o-')
            plt.title("Number of Countries with Data for All Indicators")
            plt.xlabel("Year")
            plt.ylabel("Number of Countries")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / "countries_with_data.png")
            print("Saved plot: countries_with_data.png")
        else:
            print("Insufficient yearly data for time series correlation analysis")
    else:
        print("Insufficient years with data across all indicators for time series analysis")
else:
    print("Insufficient data for correlation analysis")


# # Missing Data
