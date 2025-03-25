import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from utils.data_processing import check_normality, fig_to_base64, safe_to_datetime
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import io
import base64
import math
from scipy import stats
from matplotlib.figure import Figure
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def safe_to_datetime1(col):
    """Attempt to convert a Series to datetime, using errors='coerce'. If conversion yields all NaT, return the original column."""
    try:
        converted = pd.to_datetime(col, errors='coerce')
        # If a significant number of values were converted, return the converted series
        if converted.notna().sum() > 0:
            print(f"DEBUG: Converted column to datetime with {converted.notna().sum()} non-NaT values.")
            return converted
        else:
            print("DEBUG: Conversion resulted in all NaT, returning original column.")
            return col
    except Exception as e:
        print(f"DEBUG: safe_to_datetime conversion error: {e}")
        return col

def analyze_column(col_data: pd.Series) -> Dict[str, Any]:
    """Analyze a single column of data."""
    analysis = {'dtype': str(col_data.dtype)}

    if pd.api.types.is_numeric_dtype(col_data):
        col_data = col_data.dropna()
        is_normal, p, norm_interpret, test_name = check_normality(col_data)

        analysis.update({
            'is_normal': is_normal,
            'normality_test': test_name,
            'p_value': p,
            'normality_interpretation': norm_interpret,
            'recommended_tests': {
                'correlation': 'pearson' if is_normal else 'spearman',
                'group_comparison': 't-test' if is_normal else 'mannwhitneyu',
                'multiple_groups': 'anova' if is_normal else 'kruskal'
            },
            'recommended_tests_interpretation': {
                'correlation': 'Pearson correlation is recommended for normally distributed data to measure linear relationships. Spearman correlation is recommended for non-normal data.',
                'group_comparison': 'T-test is recommended for comparing means of two groups if data is normally distributed. Mann-Whitney U test is recommended for non-normal data.',
                'multiple_groups': 'ANOVA is recommended for comparing means of multiple groups if data is normally distributed. Kruskal-Wallis test is recommended for non-normal data.'
            }
        })

        if not is_normal:
            analysis['transform_recommendations'] = ['log', 'boxcox', 'quantile']
            analysis['transform_recommendations_interpretation'] = "Consider applying transformations like log, Box-Cox, or quantile transformation to make the data more normally distributed."

    elif pd.api.types.is_datetime64_any_dtype(col_data):
        analysis['time_analysis'] = {
            'seasonality': True,
            'trend_analysis': True
        }
        analysis['time_analysis_interpretation'] = "Time series analysis (seasonality and trend analysis) can be performed on datetime data."

    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        unique_count = len(col_data.unique())
        top_category = col_data.value_counts().index[0] if not col_data.empty else None
        analysis.update({
            'unique_values': unique_count,
            'top_category': top_category,
            'unique_values_interpretation': f"There are {unique_count} unique categories in this column.",
            'top_category_interpretation': f"The most frequent category is '{top_category}'." if top_category else "No categories to display (column is empty)."
        })

    return analysis

def calculate_stats(col_data: pd.Series) -> Dict[str, Any]:
    """Calculate statistics for a column."""
    stats_output = {}

    if pd.api.types.is_numeric_dtype(col_data):
        stats_output.update({
            'mean': col_data.mean(),
            'median': col_data.median(),
            'mode': col_data.mode()[0] if not col_data.mode().empty else None,
            'std_dev': col_data.std(),
            'variance': col_data.var(),
            'skewness': col_data.skew(),
            'kurtosis': col_data.kurtosis(),
            'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
            'min': col_data.min(),
            'max': col_data.max(),
            'missing_values': col_data.isna().sum()
        })
        
        # Add interpretations without nested braces in f-string
        stats_output.update({
            # 'mean_interpretation': f"Mean (average) value is {stats_output['mean']:.2f}.",
            # 'median_interpretation': f"Median (middle) value is {stats_output['median']:.2f}.",
            # 'mode_interpretation': (f"Mode (most frequent value) is {stats_output['mode']}." 
                                    # if stats_output['mode'] is not None 
                                    # else "No mode (all values unique or no data)."),
            # 'std_dev_interpretation': f"Standard deviation (spread) is {stats_output['std_dev']:.2f}.",
            'skewness_interpretation': f"Skewness is {stats_output['skewness']:.2f}. " + ("Right-skewed" if stats_output['skewness'] > 0 else "Left-skewed") + ".",
            'kurtosis_interpretation': f"Kurtosis is {stats_output['kurtosis']:.2f}. " + ("Heavy tails" if stats_output['kurtosis'] > 3 else "Light tails") + "."
        })

    elif pd.api.types.is_datetime64_any_dtype(col_data):
        stats_output.update({
            'min_date': col_data.min(),
            'max_date': col_data.max(),
            'unique_days': len(col_data.dt.normalize().unique()),
            'missing_values': col_data.isna().sum()
        })

    else:  # Categorical or object
        stats_output.update({
            'unique_count': col_data.nunique(),
            'top_value': col_data.value_counts().index[0] if not col_data.empty else None,
            'top_value_count': col_data.value_counts().iloc[0] if not col_data.empty else None,
            'missing_values': col_data.isna().sum()
        })

    return stats_output

def generate_graphs(col_data: pd.Series) -> Dict[str, str]:
    """Generate visualization graphs for a column."""
    graphs = {}
    graph_interpretations = {}

    if col_data.empty:
        graphs['message'] = "No data available to plot"
        graph_interpretations['message'] = "No graphs generated (empty column)."
        return {'graphs': graphs, 'interpretations': graph_interpretations}

    try:
        if pd.api.types.is_numeric_dtype(col_data):
            # Histogram with KDE
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(col_data, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col_data.name}')
            graphs['histogram'] = fig_to_base64(fig)
            graph_interpretations['histogram'] = "Histogram shows the distribution of values."

            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=col_data, ax=ax)
            ax.set_title(f'Boxplot of {col_data.name}')
            graphs['boxplot'] = fig_to_base64(fig)
            graph_interpretations['boxplot'] = "Boxplot shows quartiles and potential outliers."

            # Q-Q plot
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(col_data.dropna(), plot=ax)
            ax.set_title(f'Q-Q Plot of {col_data.name}')
            graphs['qq_plot'] = fig_to_base64(fig)
            graph_interpretations['qq_plot'] = "Q-Q plot compares data distribution to normal distribution."

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            fig, ax = plt.subplots(figsize=(12, 6))
            col_data.value_counts().sort_index().plot(ax=ax)
            ax.set_title(f'Time Distribution of {col_data.name}')
            graphs['time_density'] = fig_to_base64(fig)
            graph_interpretations['time_density'] = "Shows the distribution of events over time."

        else:  # Categorical
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = col_data.value_counts().head(10)
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Top 10 Categories in {col_data.name}')
            plt.xticks(rotation=45)
            graphs['category_distribution'] = fig_to_base64(fig)
            graph_interpretations['category_distribution'] = "Shows the frequency of top categories."

    except Exception as e:
        graphs['error'] = f"Error generating graphs: {str(e)}"
        graph_interpretations['error'] = f"Error details: {str(e)}"

    return {'graphs': graphs, 'interpretations': graph_interpretations}

def generate_pairwise_graphs(df: pd.DataFrame, col1_name: str, col2_name: str) -> Dict[str, str]:
    """Generate pairwise visualization graphs."""
    print("DEBUG: generate_pairwise_graphs called for columns:", col1_name, "and", col2_name)
    graphs = {}
    graph_interpretations = {}
    
    col1_data = df[col1_name].dropna()
    col2_data = df[col2_name].dropna()
    print("DEBUG: Column data types:", col1_data.dtype, "and", col2_data.dtype)
    
    if col1_data.empty or col2_data.empty:
        graphs['message'] = "No data available to plot for one or both columns"
        graph_interpretations['message'] = "No pairwise graphs generated (empty data)."
        return {'graphs': graphs, 'interpretations': graph_interpretations}

    try:
        if pd.api.types.is_numeric_dtype(col1_data) and pd.api.types.is_numeric_dtype(col2_data):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=col1_data, y=col2_data, ax=ax)
            ax.set_xlabel(col1_name)
            ax.set_ylabel(col2_name)
            ax.set_title(f'Scatter plot: {col1_name} vs {col2_name}')
            graphs['scatter_plot'] = fig_to_base64(fig)
            graph_interpretations['scatter_plot'] = "Scatter plot shows the relationship between variables."

        elif pd.api.types.is_numeric_dtype(col1_data) and (pd.api.types.is_categorical_dtype(col2_data) or col2_data.dtype == 'object'):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=col2_data, y=col1_data, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f'Box plot: {col1_name} by {col2_name}')
            graphs['box_plot'] = fig_to_base64(fig)
            graph_interpretations['box_plot'] = "Box plot shows distribution across categories."

        elif pd.api.types.is_numeric_dtype(col2_data) and (pd.api.types.is_categorical_dtype(col1_data) or col1_data.dtype == 'object'):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=col1_data, y=col2_data, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f'Box plot: {col2_name} by {col1_name}')
            graphs['box_plot'] = fig_to_base64(fig)
            graph_interpretations['box_plot'] = "Box plot shows distribution across categories."

        elif (pd.api.types.is_categorical_dtype(col1_data) or col1_data.dtype == 'object') and \
             (pd.api.types.is_categorical_dtype(col2_data) or col2_data.dtype == 'object'):
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.crosstab(col1_data, col2_data).plot(kind='bar', stacked=True, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f'Stacked Bar: {col1_name} vs {col2_name}')
            graphs['stacked_bar'] = fig_to_base64(fig)
            graph_interpretations['stacked_bar'] = "Stacked bar chart shows category relationships."
        else:
            print("DEBUG: No matching visualization rule for columns:", col1_name, "and", col2_name)
            graphs['message'] = "No applicable pairwise visualization available for these column types"
            graph_interpretations['message'] = f"Column types: {col1_data.dtype}, {col2_data.dtype}"
    except Exception as e:
        graphs['error'] = f"Error generating graphs: {str(e)}"
        graph_interpretations['error'] = f"Error details: {str(e)}"

    return {'graphs': graphs, 'interpretations': graph_interpretations}

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlations between variables."""
    correlations = {}
    correlation_interpretations = {}
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    standout_threshold_numerical = 0.7
    standout_threshold_p_value = 0.05

    # Numerical-Numerical correlations
    if len(numerical_cols) > 1:
        try:
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    col1, col2 = numerical_cols[i], numerical_cols[j]
                    test_type = 'spearman'
                    if check_normality(df[col1].dropna())[0] and check_normality(df[col2].dropna())[0]:
                        test_type = 'pearson'
                    
                    try:
                        if test_type == 'pearson':
                            result, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                        else:
                            result, p_value = stats.spearmanr(df[col1].dropna(), df[col2].dropna())

                        correlations[f"{col1} vs {col2}"] = {
                            'type': test_type,
                            'coefficient': result,
                            'p_value': p_value,
                            'is_standout': abs(result) > standout_threshold_numerical
                        }
                    except Exception as e:
                        correlations[f"{col1} vs {col2}"] = {"error": str(e)}
        except Exception as e:
            correlation_interpretations['numerical_matrix_error'] = f"Error in numerical correlations: {e}"

    # Add other correlation types (numerical-categorical, categorical-categorical)
    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            if df[cat_col].nunique() > 1:
                groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    try:
                        is_normal_num_col, _, _, _ = check_normality(df[num_col].dropna())
                        if is_normal_num_col:
                            stat, p = stats.f_oneway(*groups)
                            test_type = 'anova'
                        else:
                            stat, p = stats.kruskal(*groups)
                            test_type = 'kruskal-wallis'
 
                        correlations[f"{num_col} vs {cat_col}"] = {
                            'type': test_type,
                            'statistic': stat,
                            'p_value': p,
                            'is_standout': p < standout_threshold_p_value
                        }
                        correlation_interpretations[f"{num_col} vs {cat_col}"] = f"{test_type.capitalize()} test statistic = {stat:.2f}, p-value = {p:.3f}. "
                        if p < 0.05:
                            correlation_interpretations[f"{num_col} vs {cat_col}"] += f"Significant difference in '{num_col}' across categories of '{cat_col}'."
                        else:
                            correlation_interpretations[f"{num_col} vs {cat_col}"] += "No significant difference detected."
 
                    except Exception as e:
                        correlations[f"{num_col} vs {cat_col}"] = {"error": str(e)}
                        correlation_interpretations[f"{num_col} vs {cat_col}"] = f"Error computing numerical-categorical correlation: {e}"
 
    for i in range(len(categorical_cols)):
        for j in range(i+1, len(categorical_cols)):
            col1 = categorical_cols[i]
            col2 = categorical_cols[j]
            try:
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.size > 0:
                    chi2, p, _, _ = stats.chi2_contingency(contingency)
 
                    correlations[f"{col1} vs {col2}"] = {
                        'type': 'chi-squared',
                        'statistic': chi2,
                        'p_value': p,
                        'is_standout': p < standout_threshold_p_value
                    }
                    correlation_interpretations[f"{col1} vs {col2}"] = f"Chi-squared statistic = {chi2:.2f}, p-value = {p:.3f}. "
                    if p < 0.05:
                        correlation_interpretations[f"{col1} vs {col2}"] += f"Significant association between '{col1}' and '{col2}'."
                    else:
                        correlation_interpretations[f"{col1} vs {col2}"] += "No significant association detected."
            except Exception as e:
                correlations[f"{col1} vs {col2}"] = {"error": str(e)}
                correlation_interpretations[f"{col1} vs {col2}"] = f"Error computing categorical-categorical correlation: {e}"
 
    correlation_interpretations['standout_interpretation'] = (
        f"Standout correlations have coefficient > {standout_threshold_numerical} (absolute value) "
        f"or p-value < {standout_threshold_p_value}."
    )
 
    return {'correlations': correlations, 'interpretations': correlation_interpretations}

def calculate_rmf(df: pd.DataFrame, recency_col: str, frequency_col: str, monetary_col: str):
    """
    Calculates RMF scores and segments for each customer.
    
    Args:
        df: DataFrame containing customer data
        recency_col: Column representing the last transaction date/recency metric
        frequency_col: Column representing the transaction frequency (count)
        monetary_col: Column representing the monetary value (spending)
        
    Returns:
        DataFrame with RMF scores and segments
    """
    print(f"DEBUG: Starting RMF calculation with columns: {recency_col}, {frequency_col}, {monetary_col}")
    
    # Create a copy to avoid modifying the original
    rmf_df = df[[recency_col, frequency_col, monetary_col]].copy()
    
    # Handle recency properly - if it's a date, convert to days since most recent
    if pd.api.types.is_datetime64_any_dtype(rmf_df[recency_col]) or 'date' in recency_col.lower():
        try:
            print(f"DEBUG: Converting {recency_col} to datetime")
            # Try to convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(rmf_df[recency_col]):
                rmf_df[recency_col] = pd.to_datetime(rmf_df[recency_col], errors='coerce')
                
            # Get the most recent date as the reference point
            most_recent_date = rmf_df[recency_col].max()
            print(f"Reference date for recency: {most_recent_date}")
            
            # Calculate days since the most recent transaction (not from today)
            rmf_df['recency_days'] = (most_recent_date - rmf_df[recency_col]).dt.days
            
            # Log the conversion success
            print(f"DEBUG: Converted column to datetime with {rmf_df['recency_days'].notna().sum()} non-NaT values.")
            
            # Use the new column for further calculations
            recency_col = 'recency_days'
        except Exception as e:
            print(f"DEBUG: Error converting date column: {e}")
            # Keep original column if conversion fails
    
    # Make sure columns are numeric
    for col in [recency_col, frequency_col, monetary_col]:
        if not pd.api.types.is_numeric_dtype(rmf_df[col]):
            print(f"DEBUG: Converting {col} to numeric")
            # Force conversion to numeric, coercing errors to NaN
            rmf_df[col] = pd.to_numeric(rmf_df[col], errors='coerce')
    
    # ----- Recency Scoring (lower value = more recent = higher score) -----
    unique_recency = rmf_df[recency_col].dropna().nunique()
    print(f"DEBUG: Calculating R score - unique values: {unique_recency}")
    try:
        if unique_recency <= 1:
            # If there's only one value, assign the middle score
            print(f"DEBUG: Only one unique value in {recency_col}, assigning score 3")
            rmf_df['R_Score'] = 3
        else:
            # For recency, lower is better (more recent), so we invert the labels
            # We handle duplicates by dropping them in the quantile calculation
            valid_recency = rmf_df[recency_col].dropna()
            if len(valid_recency) > 5:
                # Try to use quantile-based binning first
                try:
                    # Use negative because lower values are better for recency
                    r_labels = [5, 4, 3, 2, 1]  # Reversed to make lower recency = higher score
                    r_score = pd.qcut(valid_recency, 
                                    q=5, 
                                    labels=r_labels, 
                                    duplicates='drop')
                    
                    # Map back to original dataframe
                    rmf_df['R_Score'] = pd.Series(index=valid_recency.index, data=r_score.values)
                    # Ensure integer type
                    rmf_df['R_Score'] = rmf_df['R_Score'].astype(int)
                except Exception as e:
                    print(f"DEBUG: Error with qcut for R score: {e}")
                    # Fall back to rank-based method
                    ranks = valid_recency.rank(method='min',ascending=False)
                    max_rank = ranks.max()
                    normalized = (((ranks - 1) / (max_rank - 1)) * 4 + 1).round()
                    rmf_df.loc[valid_recency.index, 'R_Score'] = normalized
                    # Ensure integer type
                    rmf_df['R_Score'] = rmf_df['R_Score'].astype(int)
            else:
                # With very few values, use simple ranking
                rmf_df['R_Score'] = pd.Series(index=valid_recency.index, 
                                           data=[5, 4, 3, 2, 1][:len(valid_recency)])
            
            # Fill NaN values with a default score of 1 (lowest)
            rmf_df['R_Score'] = rmf_df['R_Score'].fillna(1).astype(int)
    except Exception as e:
        print(f"DEBUG: Error calculating R score: {e}")
        # Catch-all fallback
        rmf_df['R_Score'] = 3  # Middle score
    
    # ----- Frequency Scoring (higher value = higher score) -----
    unique_frequency = rmf_df[frequency_col].dropna().nunique()
    print(f"DEBUG: Calculating F score - unique values: {unique_frequency}")
    try:
        if unique_frequency <= 1:
            # If there's only one value, assign the middle score
            print(f"DEBUG: Only one unique value in {frequency_col}, assigning score 3")
            rmf_df['F_Score'] = 3
        else:
            # For frequency, higher is better
            valid_frequency = rmf_df[frequency_col].dropna()
            if len(valid_frequency) > 5:
                try:
                    f_score = pd.qcut(valid_frequency, 
                                    q=5, 
                                    labels=[1, 2, 3, 4, 5], 
                                    duplicates='drop')
                    
                    # Map back to original dataframe
                    rmf_df['F_Score'] = pd.Series(index=valid_frequency.index, data=f_score.values)
                    # Ensure integer type
                    rmf_df['F_Score'] = rmf_df['F_Score'].astype(int)
                except Exception as e:
                    print(f"DEBUG: Error with qcut for F score: {e}")
                    # Fall back to rank-based method
                    ranks = valid_frequency.rank(method='min')
                    max_rank = ranks.max()
                    normalized = (((ranks - 1) / (max_rank - 1)) * 4 + 1).round()
                    rmf_df.loc[valid_frequency.index, 'F_Score'] = normalized
                    # Ensure integer type
                    rmf_df['F_Score'] = rmf_df['F_Score'].astype(int)
            else:
                # With very few values, use simple ranking
                ordered_indices = valid_frequency.sort_values().index
                scores = [1, 2, 3, 4, 5][:len(valid_frequency)]
                rmf_df.loc[ordered_indices, 'F_Score'] = scores
            
            # Fill NaN values with a default score of 1 (lowest)
            rmf_df['F_Score'] = rmf_df['F_Score'].fillna(1).astype(int)
    except Exception as e:
        print(f"DEBUG: Error calculating F score: {e}")
        # Catch-all fallback
        rmf_df['F_Score'] = 3  # Middle score
    
    # ----- Monetary Scoring (higher value = higher score) -----
    unique_monetary = rmf_df[monetary_col].dropna().nunique()
    print(f"DEBUG: Calculating M score - unique values: {unique_monetary}")
    try:
        if unique_monetary <= 1:
            # If there's only one value, assign the middle score
            print(f"DEBUG: Only one unique value in {monetary_col}, assigning score 3")
            rmf_df['M_Score'] = 3
        else:
            # For monetary, higher is better
            valid_monetary = rmf_df[monetary_col].dropna()
            if len(valid_monetary) > 5:
                try:
                    m_score = pd.qcut(valid_monetary, 
                                    q=5, 
                                    labels=[1, 2, 3, 4, 5], 
                                    duplicates='drop')
                    
                    # Map back to original dataframe
                    rmf_df['M_Score'] = pd.Series(index=valid_monetary.index, data=m_score.values)
                    # Ensure integer type
                    rmf_df['M_Score'] = rmf_df['M_Score'].astype(int)
                except Exception as e:
                    print(f"DEBUG: Error with qcut for M score: {e}")
                    # Fall back to rank-based method
                    ranks = valid_monetary.rank(method='min')
                    max_rank = ranks.max()
                    normalized = (((ranks - 1) / (max_rank - 1)) * 4 + 1).round()
                    rmf_df.loc[valid_monetary.index, 'M_Score'] = normalized
                    # Ensure integer type
                    rmf_df['M_Score'] = rmf_df['M_Score'].astype(int)
            else:
                # With very few values, use simple ranking
                ordered_indices = valid_monetary.sort_values().index
                scores = [1, 2, 3, 4, 5][:len(valid_monetary)]
                rmf_df.loc[ordered_indices, 'M_Score'] = scores
            
            # Fill NaN values with a default score of 1 (lowest)
            rmf_df['M_Score'] = rmf_df['M_Score'].fillna(1).astype(int)
    except Exception as e:
        print(f"DEBUG: Error calculating M score: {e}")
        # Catch-all fallback
        rmf_df['M_Score'] = 3  # Middle score
    
    # Ensure all scores are integers
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rmf_df[col] = rmf_df[col].astype(int)
    
    # Create RMF combined score
    rmf_df['RMF_Score'] = rmf_df['R_Score'].astype(str) + rmf_df['F_Score'].astype(str) + rmf_df['M_Score'].astype(str)
    
    # Calculate overall value
    rmf_df['RMF_Overall'] = rmf_df[['R_Score', 'F_Score', 'M_Score']].mean(axis=1)
    
    # Add segment descriptions based on RMF scores
    # Define segment descriptions based on RMF scores
    segment_descriptions = {
        '555': 'Champions',
        '554': 'Champions',
        '545': 'Champions',
        '544': 'Champions',
        '535': 'Champions',
        '455': 'Champions',
        '454': 'Champions',
        '445': 'Champions',
        '444': 'Champions',
        '435': 'Champions',
        '355': 'Champions',
        '354': 'Champions',
        '345': 'Champions',
        '344': 'Champions',
        '335': 'Champions',
        
        '543': 'Loyal Customers',
        '534': 'Loyal Customers',
        '533': 'Loyal Customers',
        '525': 'Loyal Customers',
        '524': 'Loyal Customers',
        '453': 'Loyal Customers',
        '443': 'Loyal Customers',
        '434': 'Loyal Customers',
        '425': 'Loyal Customers',
        '424': 'Loyal Customers',
        '353': 'Loyal Customers',
        '343': 'Loyal Customers',
        '334': 'Loyal Customers',
        '325': 'Loyal Customers',
        '324': 'Loyal Customers',
        
        '552': 'Potential Loyalists',
        '551': 'Potential Loyalists',
        '542': 'Potential Loyalists',
        '541': 'Potential Loyalists',
        '532': 'Potential Loyalists',
        '531': 'Potential Loyalists',
        '523': 'Potential Loyalists',
        '522': 'Potential Loyalists',
        '521': 'Potential Loyalists',
        '452': 'Potential Loyalists',
        '451': 'Potential Loyalists',
        '442': 'Potential Loyalists',
        '441': 'Potential Loyalists',
        '433': 'Potential Loyalists',
        '432': 'Potential Loyalists',
        '423': 'Potential Loyalists',
        '422': 'Potential Loyalists',
        '421': 'Potential Loyalists',
        '352': 'Potential Loyalists',
        '351': 'Potential Loyalists',
        '342': 'Potential Loyalists',
        '341': 'Potential Loyalists',
        '333': 'Potential Loyalists',
        '332': 'Potential Loyalists',
        '323': 'Potential Loyalists',
        '322': 'Potential Loyalists',
        '321': 'Potential Loyalists',
        
        '515': 'Recent Customers',
        '514': 'Recent Customers',
        '513': 'Recent Customers',
        '512': 'Recent Customers',
        '511': 'Recent Customers',
        '415': 'Recent Customers',
        '414': 'Recent Customers',
        '413': 'Recent Customers',
        '412': 'Recent Customers',
        '411': 'Recent Customers',
        '315': 'Recent Customers',
        '314': 'Recent Customers',
        '313': 'Recent Customers',
        '312': 'Recent Customers',
        '311': 'Recent Customers',
        
        '255': 'Promising',
        '254': 'Promising',
        '253': 'Promising',
        '252': 'Promising',
        '251': 'Promising',
        '245': 'Promising',
        '244': 'Promising',
        '243': 'Promising',
        '242': 'Promising',
        '241': 'Promising',
        '235': 'Promising',
        '234': 'Promising',
        '233': 'Promising',
        '232': 'Promising',
        '231': 'Promising',
        '225': 'Promising',
        '224': 'Promising',
        '223': 'Promising',
        '222': 'Promising',
        '221': 'Promising',
        '215': 'Promising',
        '214': 'Promising',
        '213': 'Promising',
        '212': 'Promising',
        '211': 'Promising',
        
        '155': 'Need Attention',
        '154': 'Need Attention',
        '153': 'Need Attention',
        '152': 'Need Attention',
        '151': 'Need Attention',
        '145': 'Need Attention',
        '144': 'Need Attention',
        '143': 'Need Attention',
        '142': 'Need Attention',
        '141': 'Need Attention',
        '135': 'Need Attention',
        '134': 'Need Attention',
        '133': 'Need Attention',
        '132': 'Need Attention',
        '131': 'Need Attention',
        '125': 'Need Attention',
        '124': 'Need Attention',
        '123': 'Need Attention',
        '122': 'Need Attention',
        '121': 'Need Attention',
        '115': 'Need Attention',
        '114': 'Need Attention',
        '113': 'Need Attention',
        '112': 'Need Attention',
        '111': 'Need Attention',
        
        # Add default for any combinations not explicitly listed
    }
    
    # Function to get segment description with a fallback
    def get_segment(rmf_code):
        # Check for specific code
        if rmf_code in segment_descriptions:
            return segment_descriptions[rmf_code]
        
        # Categorize based on R, F, M individually
        r, f, m = int(rmf_code[0]), int(rmf_code[1]), int(rmf_code[2])
        
        # Determine segment based on scores
        if r >= 4:  # Recent
            if f >= 4 and m >= 4:  # Frequent & High value
                return "Champions"
            elif f >= 3 and m >= 3:  # Moderately frequent & value
                return "Loyal Customers"
            elif f >= 2 and m >= 2:  # Less frequent or value
                return "Potential Loyalists"
            else:
                return "Recent Customers"
        elif r >= 2:  # Moderately recent
            if f >= 3 and m >= 3:  # Frequent & High value
                return "At Risk"
            elif f >= 2 and m >= 2:  # Moderately frequent & value
                return "Potential Loyalists"
            else:
                return "Promising"
        else:  # Not recent
            if f >= 4 and m >= 4:  # Frequent & High value
                return "Cannot Lose Them"
            elif f >= 3 and m >= 3:  # Moderately frequent & value
                return "At Risk"
            elif f >= 2 and m >= 2:  # Less frequent or value
                return "Hibernating"
            else:
                return "Lost"
    
    # Apply the segment descriptions
    rmf_df['Segment_Description'] = rmf_df['RMF_Score'].apply(get_segment)
    
    print(f"RMF analysis complete with {len(rmf_df)} customer records")
    return rmf_df

def calculate_rmf_kmeans(df, recency_col, frequency_col, monetary_col, clusters_range=(2, 10)):
    """
    Calculate RMF scores using K-means clustering for monetary values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with customer data
    recency_col : str
        Column name for recency (date of last transaction)
    frequency_col : str
        Column name for frequency (number of transactions)
    monetary_col : str
        Column name for monetary value (total spend)
    clusters_range : tuple, optional
        Range of clusters to test for monetary values, default is (2, 10)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RMF scores and segment descriptions
    """
    print(f"Starting advanced RMF calculation with k-means clustering")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Create a dataframe for RMF analysis
    rmf_df = pd.DataFrame()
    
    # ==== RECENCY CALCULATION ====
    print(f"Processing recency column: {recency_col}")
    # Convert date column to datetime if needed
    if df_copy[recency_col].dtype != 'datetime64[ns]':
        print(f"DEBUG (kmeans): Sample values of {recency_col} BEFORE conversion: {df_copy[recency_col].head().tolist()}") # Sample Values BEFORE
        print(f"DEBUG (kmeans): Data type of {recency_col} BEFORE conversion: {df_copy[recency_col].dtype}")
        df_copy[recency_col] = safe_to_datetime(df_copy[recency_col])
        df_copy[recency_col] = df_copy[recency_col].copy() # Re-assign to force refresh
        # print(f"DEBUG (kmeans): Sample values of {recency_col} AFTER conversion: {df_copy[recency_col].head().tolist()}") # Sample Values AFTER
        print(f"DEBUG (kmeans): Data type of {recency_col} AFTER conversion: {df_copy[recency_col].dtype}")
    
    if df_copy[recency_col].dtype == 'datetime64[ns]':
        # For recency, we want days since most recent purchase
        # Use today's date + 1 day as reference to make the most recent transaction have 1 day recency
        reference_date = df_copy[recency_col].max() + pd.Timedelta(days=1)
        print(f"Reference date for recency: {reference_date}")
        
        # Calculate days since most recent purchase (recency)
        recency_days = (reference_date - df_copy[recency_col]).dt.days
        # Add this info to our results
        rmf_df['recency_days'] = recency_days
        
        # Customers with more recent purchases should have higher scores (inverting)
        # Handle cases with only 1 unique value
        unique_recency_values = recency_days.nunique()
        print(f"Number of unique recency values: {unique_recency_values}")
        
        if unique_recency_values == 1:
            print("Only one unique recency value, assigning default score of 3")
            rmf_df['R_Score'] = 3  # Middle score if all values are the same
        elif unique_recency_values < 5:
            print(f"Few unique recency values ({unique_recency_values}), using rank-based method")
            # For few unique values, use rank-based percentile
            rmf_df['R_Score'] = pd.qcut(recency_days, q=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                                      labels=[5, 4, 3, 2, 1], duplicates='drop')
            # Convert to integer explicitly
            rmf_df['R_Score'] = rmf_df['R_Score'].astype(int)
            
            # Handle potential missing values from qcut
            if rmf_df['R_Score'].isna().any():
                # Fallback to rank method if qcut fails
                rmf_df['R_Score'] = 5 - (recency_days.rank(method='first', 
                                                          pct=True) * 4).astype(int) + 1
        else:
            # Convert qcut result to integer explicitly before arithmetic
            temp_qcut = pd.qcut(recency_days, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            temp_qcut = temp_qcut.astype(int)  # Convert to integer before arithmetic

            # Normal case with many unique values - higher score for lower recency
            rmf_df['R_Score'] = 6 - temp_qcut  # Use 6 minus to invert (5 becomes 1, 1 becomes 5)
            
            # Handle potential missing values from qcut
            if rmf_df['R_Score'].isna().any():
                print("Handling missing R_Scores with rank method")
                # Fallback to rank method if qcut fails
                rmf_df['R_Score'] = 5 - (recency_days.rank(method='first', 
                                                          pct=True) * 4).astype(int) + 1
    else:
        print(f"Recency column {recency_col} is not a date, treating as numeric")
        # If not a date, treat as a numeric column where lower values are more recent
        recency_values = df_copy[recency_col]
        rmf_df['recency_days'] = recency_values
        
        unique_recency_values = recency_values.nunique()
        print(f"Number of unique recency values: {unique_recency_values}")
        
        if unique_recency_values == 1:
            print("Only one unique recency value, assigning default score of 3")
            rmf_df['R_Score'] = 3  # Middle score if all values are the same
        elif unique_recency_values < 5:
            print(f"Few unique recency values ({unique_recency_values}), using rank-based method")
            # For few unique values, use rank-based percentile
            rmf_df['R_Score'] = pd.qcut(recency_values, q=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                                      labels=[5, 4, 3, 2, 1], duplicates='drop')
            # Convert to integer explicitly
            rmf_df['R_Score'] = rmf_df['R_Score'].astype(int)
            
            # Handle potential missing values from qcut
            if rmf_df['R_Score'].isna().any():
                # Fallback to rank method if qcut fails
                rmf_df['R_Score'] = 5 - (recency_values.rank(method='first', 
                                                           pct=True) * 4).astype(int) + 1
        else:
            # Normal case with many unique values - higher score for lower recency (invert)
            rmf_df['R_Score'] = 5 - pd.qcut(recency_values, q=5, 
                                          labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Handle potential missing values from qcut
            if rmf_df['R_Score'].isna().any():
                print("Handling missing R_Scores with rank method")
                rmf_df['R_Score'] = 5 - (recency_values.rank(method='first', 
                                                           pct=True) * 4).astype(int) + 1
    
    # ==== FREQUENCY CALCULATION ====
    print(f"Processing frequency column: {frequency_col}")
    freq_values = df_copy[frequency_col]
    
    # Handle missing values in frequency
    if freq_values.isna().any():
        print(f"Warning: {freq_values.isna().sum()} missing values in frequency column")
        freq_values = freq_values.fillna(0)  # Replace missing with zeros for frequency
    
    # Check if frequency is numeric
    if not pd.api.types.is_numeric_dtype(freq_values):
        try:
            print(f"Converting frequency column to numeric")
            freq_values = pd.to_numeric(freq_values, errors='coerce')
            freq_values = freq_values.fillna(0)
        except:
            print(f"Could not convert frequency column to numeric, using rank-based method")
            rmf_df['F_Score'] = (freq_values.rank(method='first', 
                                                pct=True) * 4).astype(int)
            rmf_df['F_Score'] = rmf_df['F_Score'].fillna(1).astype(int)
            
    unique_freq_values = freq_values.nunique()
    print(f"Number of unique frequency values: {unique_freq_values}")
    
    if unique_freq_values == 1:
        print("Only one unique frequency value, assigning default score of 3")
        rmf_df['F_Score'] = 3  # Middle score if all values are the same
    elif unique_freq_values < 5:
        print(f"Few unique frequency values ({unique_freq_values}), using rank-based method")
        # For few unique values, use rank-based percentile
        temp_qcut = pd.qcut(freq_values, q=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                           labels=[1, 2, 3, 4, 5], duplicates='drop')
        # Convert to integer explicitly
        rmf_df['F_Score'] = temp_qcut.astype(int)
        
        # Handle potential missing values from qcut
        if rmf_df['F_Score'].isna().any():
            # Fallback to rank method if qcut fails
            rmf_df['F_Score'] = (freq_values.rank(method='first', 
                                                pct=True) * 4).astype(int) + 1
    elif unique_freq_values == 5:
        # Special handling for exactly 5 unique values - direct mapping
        print("Exactly 5 unique frequency values - using direct score mapping")
        # Sort the unique values
        unique_values = sorted(freq_values.unique())
        # Create mapping from values to scores 1-5
        value_to_score = {val: score for val, score in zip(unique_values, range(1, 6))}
        print(f"DEBUG: Frequency value to score mapping: {value_to_score}")
        # Apply mapping
        rmf_df['F_Score'] = freq_values.map(value_to_score).astype(int)
    else:
        # Normal case with many unique values - higher score for higher frequency
        # Convert qcut result to integer explicitly before any arithmetic
        try:
            temp_qcut = pd.qcut(freq_values, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rmf_df['F_Score'] = temp_qcut.astype(int)
        except Exception as e:
            print(f"Error with qcut for frequency: {e}")
            print("Falling back to rank-based scoring for frequency")
            # Fallback to rank-based scoring
            rmf_df['F_Score'] = (freq_values.rank(method='first', pct=True) * 4 + 1).astype(int)
        
        # Handle potential missing values from qcut
        if rmf_df['F_Score'].isna().any():
            print("Handling missing F_Scores with rank method")
            rmf_df['F_Score'] = (freq_values.rank(method='first', 
                                                pct=True) * 4).astype(int) + 1
    
    # ==== MONETARY CALCULATION WITH K-MEANS ====
    print(f"Processing monetary column: {monetary_col}")
    monetary_values = df_copy[monetary_col]
    
    # Handle missing values in monetary
    if monetary_values.isna().any():
        print(f"Warning: {monetary_values.isna().sum()} missing values in monetary column")
        monetary_values = monetary_values.fillna(0)  # Replace missing with zeros for monetary
    
    # Check if monetary is numeric
    if not pd.api.types.is_numeric_dtype(monetary_values):
        try:
            print(f"Converting monetary column to numeric")
            monetary_values = pd.to_numeric(monetary_values, errors='coerce')
            monetary_values = monetary_values.fillna(0)
        except:
            print(f"Could not convert monetary column to numeric, using rank-based method")
            rmf_df['M_Score'] = (monetary_values.rank(method='first', 
                                                    pct=True) * 4).astype(int)
            rmf_df['M_Score'] = rmf_df['M_Score'].fillna(1).astype(int)

    # Handle special cases
    unique_monetary_values = monetary_values.nunique()
    print(f"Number of unique monetary values: {unique_monetary_values}")
    
    if unique_monetary_values == 1:
        print("Only one unique monetary value, assigning default score of 3")
        rmf_df['M_Score'] = 3  # Middle score if all values are the same
        rmf_df['M_Cluster'] = 1  # Single cluster
    elif unique_monetary_values < 5:
        print(f"Few unique monetary values ({unique_monetary_values}), using quantile method")
        # For few unique values, use quantile-based scoring
        temp_qcut = pd.qcut(monetary_values, q=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                           labels=[1, 2, 3, 4, 5], duplicates='drop')
        # Convert to integer explicitly 
        rmf_df['M_Score'] = temp_qcut.astype(int)
        rmf_df['M_Cluster'] = rmf_df['M_Score']  # Use score as cluster
        
        # Handle potential missing values from qcut
        if rmf_df['M_Score'].isna().any():
            # Fallback to rank method if qcut fails
            rmf_df['M_Score'] = (monetary_values.rank(method='first', 
                                                    pct=True) * 4).astype(int) + 1
            rmf_df['M_Cluster'] = rmf_df['M_Score']
    else:
        # If we have enough unique values, use K-means clustering
        # Apply log transformation to handle skewness (with offset to handle zeros/negatives)
        print("Using K-means clustering for monetary values")
        
        # Get the minimum value to ensure all values are positive for log transform
        min_value = monetary_values.min()
        offset = 0
        if min_value <= 0:
            offset = abs(min_value) + 1  # Add 1 to avoid log(0)
        
        # Log transform the data (with offset if needed)
        try:
            log_monetary = np.log1p(monetary_values + offset)
            
            # Scale the data using RobustScaler to handle outliers
            scaler = RobustScaler()
            scaled_log_monetary = scaler.fit_transform(log_monetary.values.reshape(-1, 1))
            
            # Determine optimal number of clusters using elbow method
            inertia_values = []
            min_clusters, max_clusters = clusters_range
            max_clusters = min(max_clusters, unique_monetary_values - 1)
            
            # Ensure we have a valid range
            min_clusters = max(2, min_clusters)
            max_clusters = max(min_clusters + 1, max_clusters)
            
            print(f"Testing cluster range: {min_clusters} to {max_clusters}")
            
            # Calculate inertia for each number of clusters
            for k in range(min_clusters, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_log_monetary)
                    inertia_values.append(kmeans.inertia_)
                    print(f"K={k}, Inertia: {kmeans.inertia_}")
                except Exception as e:
                    print(f"Error with K={k}: {e}")
                    # Skip this k value
            
            # Calculate inertia differences to find the elbow point
            if len(inertia_values) > 1:
                inertia_diffs = np.diff(inertia_values)
                inertia_diffs_percent = inertia_diffs / inertia_values[:-1]
                
                # Find the elbow point - where the rate of change decreases significantly
                # Use a simple threshold method
                optimal_k = min_clusters  # Default
                for i, diff_pct in enumerate(inertia_diffs_percent):
                    if abs(diff_pct) < 0.2:  # If improvement is less than 20%
                        optimal_k = min_clusters + i + 1
                        break
                
                # Fallback if we didn't find a clear elbow
                if optimal_k == min_clusters and len(inertia_diffs_percent) > 0:
                    # Find the point with the biggest change
                    optimal_idx = np.argmax(np.abs(inertia_diffs_percent))
                    optimal_k = min_clusters + optimal_idx + 1
                
                print(f"Optimal number of clusters: {optimal_k}")
            else:
                # Not enough values to determine optimal k
                optimal_k = min_clusters
                print(f"Using default number of clusters: {optimal_k}")
            
            # Apply K-means clustering with the optimal number of clusters
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_log_monetary)
            
            # Get cluster centers and sort them to assign proper scores
            centers = kmeans.cluster_centers_.flatten()
            cluster_ranks = np.argsort(centers)
            
            # Create a mapping from cluster to rank (score)
            cluster_to_score = {}
            for rank, cluster in enumerate(cluster_ranks):
                # Map to scores 1-5
                score = int(np.floor(rank * 5 / len(cluster_ranks))) + 1
                # Ensure we don't exceed 5
                score = min(score, 5)
                cluster_to_score[cluster] = score
            
            # Store the original cluster assignments
            rmf_df['M_Cluster'] = clusters + 1  # Add 1 for 1-based indexing
            
            # Map clusters to scores (as integers)
            rmf_df['M_Score'] = pd.Series(clusters).map(cluster_to_score).astype(int)
            
            # Ensure we have 5 score levels max by quantizing if needed
            if len(set(rmf_df['M_Score'])) > 5:
                print("Rescaling M_Score to 5 levels")
                # Use qcut to remap to 5 levels
                temp_scores = pd.qcut(rmf_df['M_Score'], q=5, 
                                   labels=[1, 2, 3, 4, 5], duplicates='drop')
                # Convert to integer explicitly
                rmf_df['M_Score'] = temp_scores.astype(int)
        except Exception as e:
            print(f"Error in K-means clustering: {e}")
            print("Falling back to quantile-based scoring for monetary values")
            # Fallback to regular quantile-based scoring
            temp_scores = pd.qcut(monetary_values, q=5, 
                               labels=[1, 2, 3, 4, 5], duplicates='drop')
            # Convert to integer explicitly
            rmf_df['M_Score'] = temp_scores.astype(int)
            
            # Handle potential missing values from qcut
            if rmf_df['M_Score'].isna().any():
                print("Handling missing M_Scores with rank method")
                # Fallback to rank method if qcut fails
                rmf_df['M_Score'] = (monetary_values.rank(method='first', 
                                                        pct=True) * 4).astype(int) + 1
    
    # Ensure all scores are integers
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rmf_df[col] = rmf_df[col].astype(int)
    
    # Create RMF combined score
    rmf_df['RMF_Score'] = rmf_df['R_Score'].astype(str) + rmf_df['F_Score'].astype(str) + rmf_df['M_Score'].astype(str)
    
    # Calculate overall value
    rmf_df['RMF_Overall'] = rmf_df[['R_Score', 'F_Score', 'M_Score']].mean(axis=1)
    
    # Add segment descriptions based on RMF scores
    # Define segment descriptions based on RMF scores
    segment_descriptions = {
        '555': 'Champions',
        '554': 'Champions',
        '545': 'Champions',
        '544': 'Champions',
        '535': 'Champions',
        '455': 'Champions',
        '454': 'Champions',
        '445': 'Champions',
        '444': 'Champions',
        '435': 'Champions',
        '355': 'Champions',
        '354': 'Champions',
        '345': 'Champions',
        '344': 'Champions',
        '335': 'Champions',
        
        '543': 'Loyal Customers',
        '534': 'Loyal Customers',
        '533': 'Loyal Customers',
        '525': 'Loyal Customers',
        '524': 'Loyal Customers',
        '453': 'Loyal Customers',
        '443': 'Loyal Customers',
        '434': 'Loyal Customers',
        '425': 'Loyal Customers',
        '424': 'Loyal Customers',
        '353': 'Loyal Customers',
        '343': 'Loyal Customers',
        '334': 'Loyal Customers',
        '325': 'Loyal Customers',
        '324': 'Loyal Customers',
        
        '552': 'Potential Loyalists',
        '551': 'Potential Loyalists',
        '542': 'Potential Loyalists',
        '541': 'Potential Loyalists',
        '532': 'Potential Loyalists',
        '531': 'Potential Loyalists',
        '523': 'Potential Loyalists',
        '522': 'Potential Loyalists',
        '521': 'Potential Loyalists',
        '452': 'Potential Loyalists',
        '451': 'Potential Loyalists',
        '442': 'Potential Loyalists',
        '441': 'Potential Loyalists',
        '433': 'Potential Loyalists',
        '432': 'Potential Loyalists',
        '423': 'Potential Loyalists',
        '422': 'Potential Loyalists',
        '421': 'Potential Loyalists',
        '352': 'Potential Loyalists',
        '351': 'Potential Loyalists',
        '342': 'Potential Loyalists',
        '341': 'Potential Loyalists',
        '333': 'Potential Loyalists',
        '332': 'Potential Loyalists',
        '323': 'Potential Loyalists',
        '322': 'Potential Loyalists',
        '321': 'Potential Loyalists',
        
        '515': 'Recent Customers',
        '514': 'Recent Customers',
        '513': 'Recent Customers',
        '512': 'Recent Customers',
        '511': 'Recent Customers',
        '415': 'Recent Customers',
        '414': 'Recent Customers',
        '413': 'Recent Customers',
        '412': 'Recent Customers',
        '411': 'Recent Customers',
        '315': 'Recent Customers',
        '314': 'Recent Customers',
        '313': 'Recent Customers',
        '312': 'Recent Customers',
        '311': 'Recent Customers',
        
        '255': 'Promising',
        '254': 'Promising',
        '253': 'Promising',
        '252': 'Promising',
        '251': 'Promising',
        '245': 'Promising',
        '244': 'Promising',
        '243': 'Promising',
        '242': 'Promising',
        '241': 'Promising',
        '235': 'Promising',
        '234': 'Promising',
        '233': 'Promising',
        '232': 'Promising',
        '231': 'Promising',
        '225': 'Promising',
        '224': 'Promising',
        '223': 'Promising',
        '222': 'Promising',
        '221': 'Promising',
        '215': 'Promising',
        '214': 'Promising',
        '213': 'Promising',
        '212': 'Promising',
        '211': 'Promising',
        
        '155': 'Need Attention',
        '154': 'Need Attention',
        '153': 'Need Attention',
        '152': 'Need Attention',
        '151': 'Need Attention',
        '145': 'Need Attention',
        '144': 'Need Attention',
        '143': 'Need Attention',
        '142': 'Need Attention',
        '141': 'Need Attention',
        '135': 'Need Attention',
        '134': 'Need Attention',
        '133': 'Need Attention',
        '132': 'Need Attention',
        '131': 'Need Attention',
        '125': 'Need Attention',
        '124': 'Need Attention',
        '123': 'Need Attention',
        '122': 'Need Attention',
        '121': 'Need Attention',
        '115': 'Need Attention',
        '114': 'Need Attention',
        '113': 'Need Attention',
        '112': 'Need Attention',
        '111': 'Need Attention',
        
        # Add default for any combinations not explicitly listed
    }
    
    # Function to get segment description with a fallback
    def get_segment(rmf_code):
        # Check for specific code
        if rmf_code in segment_descriptions:
            return segment_descriptions[rmf_code]
        
        # Categorize based on R, F, M individually
        r, f, m = int(rmf_code[0]), int(rmf_code[1]), int(rmf_code[2])
        
        # Determine segment based on scores
        if r >= 4:  # Recent
            if f >= 4 and m >= 4:  # Frequent & High value
                return "Champions"
            elif f >= 3 and m >= 3:  # Moderately frequent & value
                return "Loyal Customers"
            elif f >= 2 and m >= 2:  # Less frequent or value
                return "Potential Loyalists"
            else:
                return "Recent Customers"
        elif r >= 2:  # Moderately recent
            if f >= 3 and m >= 3:  # Frequent & High value
                return "At Risk"
            elif f >= 2 and m >= 2:  # Moderately frequent & value
                return "Potential Loyalists"
            else:
                return "Promising"
        else:  # Not recent
            if f >= 4 and m >= 4:  # Frequent & High value
                return "Cannot Lose Them"
            elif f >= 3 and m >= 3:  # Moderately frequent & value
                return "At Risk"
            elif f >= 2 and m >= 2:  # Less frequent or value
                return "Hibernating"
            else:
                return "Lost"
    
    # Apply the segment descriptions
    rmf_df['Segment_Description'] = rmf_df['RMF_Score'].apply(get_segment)
    
    print(f"RMF analysis complete with {len(rmf_df)} customer records")
    return rmf_df
