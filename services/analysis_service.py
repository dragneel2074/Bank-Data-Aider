import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from utils.data_processing import check_normality, fig_to_base64
import scipy.stats as stats

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
            'mean_interpretation': f"Mean (average) value is {stats_output['mean']:.2f}.",
            'median_interpretation': f"Median (middle) value is {stats_output['median']:.2f}.",
            'mode_interpretation': (f"Mode (most frequent value) is {stats_output['mode']}." 
                                    if stats_output['mode'] is not None 
                                    else "No mode (all values unique or no data)."),
            'std_dev_interpretation': f"Standard deviation (spread) is {stats_output['std_dev']:.2f}.",
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
    
    # Make sure columns are numeric
    for col in [recency_col, frequency_col, monetary_col]:
        if not pd.api.types.is_numeric_dtype(rmf_df[col]):
            print(f"DEBUG: Converting {col} to numeric")
            rmf_df[col] = pd.to_numeric(rmf_df[col], errors='coerce')
    
    # Handle Recency - Lower values are better (more recent)
    r_labels = range(5, 0, -1)  # 5 for most recent, 1 for least recent
    try:
        rmf_df['R_Score'] = pd.qcut(rmf_df[recency_col], q=5, labels=r_labels, duplicates='drop')
    except ValueError as e:
        print(f"DEBUG: Error in R quantile calculation: {e}")
        # Handle case where there aren't enough unique values
        rmf_df['R_Score'] = pd.cut(rmf_df[recency_col], bins=5, labels=r_labels, duplicates='drop')
    
    # Handle Frequency - Higher values are better
    f_labels = range(1, 6)  # 1 for lowest frequency, 5 for highest
    try:
        rmf_df['F_Score'] = pd.qcut(rmf_df[frequency_col], q=5, labels=f_labels, duplicates='drop')
    except ValueError as e:
        print(f"DEBUG: Error in F quantile calculation: {e}")
        # Handle case where there aren't enough unique values
        rmf_df['F_Score'] = pd.cut(rmf_df[frequency_col], bins=5, labels=f_labels, duplicates='drop')
    
    # Handle Monetary Value - Higher values are better
    m_labels = range(1, 6)  # 1 for lowest monetary, 5 for highest
    try:
        rmf_df['M_Score'] = pd.qcut(rmf_df[monetary_col], q=5, labels=m_labels, duplicates='drop')
    except ValueError as e:
        print(f"DEBUG: Error in M quantile calculation: {e}")
        # Handle case where there aren't enough unique values
        rmf_df['M_Score'] = pd.cut(rmf_df[monetary_col], bins=5, labels=m_labels, duplicates='drop')
    
    # Convert score columns to numeric for calculations
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rmf_df[col] = pd.to_numeric(rmf_df[col], errors='coerce')
    
    # Combine RFM scores
    rmf_df['RMF_Segment'] = rmf_df['R_Score'].astype(str) + rmf_df['F_Score'].astype(str) + rmf_df['M_Score'].astype(str)
    rmf_df['RMF_Score'] = rmf_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # Add segment descriptions
    segments = {
        'Champions': rmf_df[(rmf_df['R_Score'] >= 4) & (rmf_df['F_Score'] >= 4) & (rmf_df['M_Score'] >= 4)].index,
        'Loyal Customers': rmf_df[(rmf_df['R_Score'] >= 3) & (rmf_df['F_Score'] >= 3) & (rmf_df['M_Score'] >= 3)].index,
        'Potential Loyalists': rmf_df[(rmf_df['R_Score'] >= 3) & (rmf_df['F_Score'] >= 2) & (rmf_df['M_Score'] >= 2)].index,
        'New Customers': rmf_df[(rmf_df['R_Score'] >= 4) & (rmf_df['F_Score'] <= 2)].index,
        'Promising': rmf_df[(rmf_df['R_Score'] >= 3) & (rmf_df['F_Score'] <= 2) & (rmf_df['M_Score'] <= 2)].index,
        'Needs Attention': rmf_df[(rmf_df['R_Score'] >= 2) & (rmf_df['F_Score'] >= 2) & (rmf_df['M_Score'] >= 2)].index,
        'About to Sleep': rmf_df[(rmf_df['R_Score'] == 2) & (rmf_df['F_Score'] <= 2) & (rmf_df['M_Score'] <= 2)].index,
        'At Risk': rmf_df[(rmf_df['R_Score'] <= 2) & (rmf_df['F_Score'] >= 3) & (rmf_df['M_Score'] >= 3)].index,
        'Cannot Lose Them': rmf_df[(rmf_df['R_Score'] <= 1) & (rmf_df['F_Score'] >= 4) & (rmf_df['M_Score'] >= 4)].index,
        'Hibernating': rmf_df[(rmf_df['R_Score'] <= 2) & (rmf_df['F_Score'] <= 2)].index,
        'Lost': rmf_df[(rmf_df['R_Score'] <= 1) & (rmf_df['F_Score'] <= 1)].index
    }
    
    rmf_df['Segment_Description'] = 'Other'
    for segment, idx in segments.items():
        rmf_df.loc[idx, 'Segment_Description'] = segment
    
    print(f"DEBUG: RMF calculation complete, generated {len(rmf_df)} scores")
    return rmf_df 