from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Any

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper functions
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def check_normality(data: pd.Series) -> tuple:
    if len(data) < 3:
        return False, None, "Not enough data points to perform normality test.", None
    try:
        if len(data) > 5000:
            stat, p = stats.kstest(data, 'norm')
            test_name = 'Kolmogorov-Smirnov'
        else:
            stat, p = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
        is_normal = p > 0.05
        interpretation = f"{test_name} Test p-value = {p:.3f}. "
        if is_normal:
            interpretation += "Data is likely normally distributed (fail to reject null hypothesis)."
        else:
            interpretation += "Data is NOT normally distributed (reject null hypothesis)."
        return is_normal, p, interpretation, test_name
    except Exception as e:
        return False, None, f"Normality test could not be performed: {e}", None

def convert_numpy_types(obj):
    """Recursively convert numpy data types to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

def analyze_column(col_data: pd.Series) -> Dict[str, Any]:
    analysis = {'dtype': str(col_data.dtype)}

    if pd.api.types.is_numeric_dtype(col_data):
        # Handle numerical data
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
                'correlation': 'Pearson correlation is recommended for normally distributed data to measure linear relationships. Spearman correlation is recommended for non-normal data to measure monotonic relationships.',
                'group_comparison': 'T-test is recommended for comparing means of two groups if data is normally distributed. Mann-Whitney U test is recommended for non-normal data.',
                'multiple_groups': 'ANOVA is recommended for comparing means of multiple groups if data is normally distributed. Kruskal-Wallis test is recommended for non-normal data.'
            }
        })

        if not is_normal:
            analysis['transform_recommendations'] = ['log', 'boxcox', 'quantile']
            analysis['transform_recommendations_interpretation'] = "Consider applying transformations like log, Box-Cox, or quantile transformation to make the data more normally distributed, which might be beneficial for some statistical tests that assume normality."

    elif pd.api.types.is_datetime64_any_dtype(col_data):
        # Handle datetime data
        analysis['time_analysis'] = {
            'seasonality': True,
            'trend_analysis': True
        }
        analysis['time_analysis_interpretation'] = "Time series analysis (seasonality and trend analysis) can be performed on datetime data to understand temporal patterns."

    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        # Handle categorical data
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
    stats_output = {}

    if pd.api.types.is_numeric_dtype(col_data):
        mean_val = col_data.mean()
        median_val = col_data.median()
        mode_val = col_data.mode()[0] if not col_data.mode().empty else None
        std_dev_val = col_data.std()
        variance_val = col_data.var()
        skewness_val = col_data.skew()
        kurtosis_val = col_data.kurtosis()
        iqr_val = col_data.quantile(0.75) - col_data.quantile(0.25)
        min_val = col_data.min()
        max_val = col_data.max()
        missing_values_count = col_data.isna().sum()

        stats_output = {
            'mean': mean_val,
            'median': median_val,
            'mode': mode_val,
            'std_dev': std_dev_val,
            'variance': variance_val,
            'skewness': skewness_val,
            'kurtosis': kurtosis_val,
            'iqr': iqr_val,
            'min': min_val,
            'max': max_val,
            'missing_values': missing_values_count,
            'mean_interpretation': f"Mean (average) value is {mean_val:.2f}.",
            'median_interpretation': f"Median (middle) value is {median_val:.2f}. Less sensitive to outliers than mean.",
            'mode_interpretation': f"Mode (most frequent value) is {mode_val}." if mode_val is not None else "No mode (all values unique or no data).",
            'std_dev_interpretation': f"Standard deviation (data spread) is {std_dev_val:.2f}. Higher value means greater spread.",
            'variance_interpretation': f"Variance (squared spread) is {variance_val:.2f}.",
            'skewness_interpretation': f"Skewness (distribution symmetry) is {skewness_val:.2f}. 0=symmetric, >0=right-skewed, <0=left-skewed.",
            'kurtosis_interpretation': f"Kurtosis (tail heaviness) is {kurtosis_val:.2f}. 3=normal, >3=heavy tails, <3=light tails.",
            'iqr_interpretation': f"Interquartile Range (middle 50% range) is {iqr_val:.2f}. Robust measure of spread.",
            'min_interpretation': f"Minimum value is {min_val:.2f}.",
            'max_interpretation': f"Maximum value is {max_val:.2f}.",
            'missing_values_interpretation': f"Number of missing values: {missing_values_count}."
        }
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        min_date_val = col_data.min()
        max_date_val = col_data.max()
        unique_days_count = len(col_data.dt.normalize().unique())
        missing_values_count = col_data.isna().sum()

        stats_output = {
            'min_date': min_date_val.isoformat() if not pd.isna(min_date_val) else None,
            'max_date': max_date_val.isoformat() if not pd.isna(max_date_val) else None,
            'unique_days': unique_days_count,
            'missing_values': missing_values_count,
            'min_date_interpretation': f"Earliest date is {min_date_val.isoformat()[:10]}." if not pd.isna(min_date_val) else "No date data available.",
            'max_date_interpretation': f"Latest date is {max_date_val.isoformat()[:10]}." if not pd.isna(max_date_val) else "No date data available.",
            'unique_days_interpretation': f"Number of unique days with data: {unique_days_count}.",
            'missing_values_interpretation': f"Number of missing date values: {missing_values_count}."
        }
    else:
        unique_count_val = col_data.nunique()
        top_value_val = col_data.value_counts().index[0] if not col_data.empty else None
        top_value_count_val = col_data.value_counts().iloc[0] if not col_data.empty else None
        missing_values_count = col_data.isna().sum()

        stats_output = {
            'unique_count': unique_count_val,
            'top_value': top_value_val,
            'top_value_count': top_value_count_val,
            'missing_values': missing_values_count,
            'unique_count_interpretation': f"Number of unique values: {unique_count_val}.",
            'top_value_interpretation': f"Most frequent value is '{top_value_val}'.",
            'top_value_count_interpretation': f"Frequency of the most frequent value: {top_value_count_val}.",
            'missing_values_interpretation': f"Number of missing values: {missing_values_count}."
        }

    return stats_output

def generate_graphs(col_data: pd.Series) -> Dict[str, str]:
    graphs = {}
    graph_interpretations = {}

    if col_data.empty:
        graphs['message'] = "No data available to plot"
        graph_interpretations['message'] = "No graphs generated as the column contains no data."
        return {'graphs': graphs, 'interpretations': graph_interpretations}

    if pd.api.types.is_numeric_dtype(col_data):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(col_data, kde=True, ax=ax)
            graphs['histogram'] = fig_to_base64(fig)
            graph_interpretations['histogram'] = "Histogram shows the distribution of numerical data, bars represent frequency of values in bins. KDE line provides a smooth estimate of the probability density."
        except Exception as e:
            graphs['histogram'] = "Error generating histogram"
            graph_interpretations['histogram'] = f"Error generating histogram: {e}"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=col_data, ax=ax)
            graphs['boxplot'] = fig_to_base64(fig)
            graph_interpretations['boxplot'] = "Boxplot displays the distribution through quartiles, median, and potential outliers. Useful for comparing distributions and identifying skewness."
        except Exception as e:
            graphs['boxplot'] = "Error generating boxplot"
            graph_interpretations['boxplot'] = f"Error generating boxplot: {e}"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(col_data.dropna(), plot=ax)
            graphs['qq_plot'] = fig_to_base64(fig)
            graph_interpretations['qq_plot'] = "Q-Q plot (Quantile-Quantile plot) compares data distribution to a normal distribution. Points close to the red line suggest normality."
        except Exception as e:
            graphs['qq_plot'] = "Error generating Q-Q plot"
            graph_interpretations['qq_plot'] = f"Error generating Q-Q plot: {e}"
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            col_data.groupby(col_data.dt.floor('d')).count().plot(ax=ax)
            ax.set_title('Transaction Density Over Time')
            graphs['time_density'] = fig_to_base64(fig)
            graph_interpretations['time_density'] = "Time density plot shows the frequency of data points over time. Useful for visualizing trends and patterns over time."
        except Exception as e:
            graphs['time_density'] = "Error generating time density plot"
            graph_interpretations['time_density'] = f"Error generating time density plot: {e}"
        # try: # Re-enable seasonality decomposition if needed, might be computationally intensive for large datasets
        #     ts_series = col_data.dropna().resample('D').count()
        #     if len(ts_series) >= 60:
        #         result = seasonal_decompose(ts_series, period=30, model='additive', extrapolate_trend='freq') # Added model and extrapolate_trend
        #         fig = result.plot()
        #         graphs['seasonality'] = fig_to_base64(fig)
        #         graph_interpretations['seasonality'] = "Seasonal decomposition plot breaks down the time series into trend, seasonal, and residual components to understand underlying patterns."
        #     else:
        #         graphs['seasonality'] = "Insufficient data for seasonal decomposition"
        #         graph_interpretations['seasonality'] = "Seasonal decomposition requires sufficient data points over time. Not enough data to perform decomposition."
        # except Exception as e:
        #     graphs['seasonality'] = "Error generating seasonal decomposition plot"
        #     graph_interpretations['seasonality'] = f"Error generating seasonal decomposition plot: {e}"
    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            col_data.value_counts().head(10).plot(kind='bar', ax=ax) # Limit to top 10 categories for better visualization
            graphs['category_distribution'] = fig_to_base64(fig)
            graph_interpretations['category_distribution'] = "Bar chart showing the distribution of the top categories. Useful for understanding the frequency of different categories."
        except Exception as e:
            graphs['category_distribution'] = "Error generating category distribution plot"
            graph_interpretations['category_distribution'] = f"Error generating category distribution plot: {e}"

    return {'graphs': graphs, 'interpretations': graph_interpretations}

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    correlations = {}
    correlation_interpretations = {}
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    standout_threshold_numerical = 0.7  # Threshold for strong numerical correlation
    standout_threshold_p_value = 0.05 # Threshold for significant p-value

    # Numerical-Numerical correlations
    if len(numerical_cols) > 1:
        try:
            corr_matrix = df[numerical_cols].corr(method='spearman') # Default to Spearman for robustness
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    try:
                        test_type = 'spearman'
                        if check_normality(df[col1].dropna())[0] and check_normality(df[col2].dropna())[0]:
                            test_type = 'pearson' # Use Pearson if both are normal
                        if test_type == 'pearson':
                            result, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                        else: # Spearman
                            result, p_value = stats.spearmanr(df[col1].dropna(), df[col2].dropna())

                        is_standout = False
                        if abs(result) > standout_threshold_numerical:
                            is_standout = True

                        correlations[f"{col1} vs {col2}"] = {
                            'type': test_type,
                            'coefficient': result, # Use the direct result from pearsonr/spearmanr
                            'p_value': p_value,
                            'is_standout': is_standout # Add is_standout flag
                        }
                        correlation_interpretations[f"{col1} vs {col2}"] = f"{test_type.capitalize()} correlation coefficient between '{col1}' and '{col2}' is {result:.2f} (p-value={p_value:.3f}). "
                        if abs(result) > 0.7:
                            correlation_interpretations[f"{col1} vs {col2}"] += "Strong correlation."
                        elif abs(result) > 0.3:
                            correlation_interpretations[f"{col1} vs {col2}"] += "Moderate correlation."
                        else:
                            correlation_interpretations[f"{col1} vs {col2}"] += "Weak or no correlation."

                    except Exception as e:
                        correlations[f"{col1} vs {col2}"] = {"error": str(e)}
                        correlation_interpretations[f"{col1} vs {col2}"] = f"Error computing numerical correlation: {e}"
        except Exception as e:
            correlation_interpretations['numerical_matrix_error'] = f"Error computing numerical correlation matrix: {e}"

    # Numerical-Categorical correlations
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

                        is_standout = False
                        if p < standout_threshold_p_value:
                            is_standout = True

                        correlations[f"{num_col} vs {cat_col}"] = {
                            'type': test_type,
                            'statistic': stat,
                            'p_value': p,
                            'is_standout': is_standout # Add is_standout flag
                        }
                        correlation_interpretations[f"{num_col} vs {cat_col}"] = f"{test_type.capitalize()} test statistic = {stat:.2f}, p-value = {p:.3f}. "
                        if p < 0.05:
                            correlation_interpretations[f"{num_col} vs {cat_col}"] += f"Significant difference in '{num_col}' across categories of '{cat_col}'."
                        else:
                            correlation_interpretations[f"{num_col} vs {cat_col}"] += "No significant difference detected."

                    except Exception as e:
                        correlations[f"{num_col} vs {cat_col}"] = {"error": str(e)}
                        correlation_interpretations[f"{num_col} vs {cat_col}"] = f"Error computing numerical-categorical correlation: {e}"

    # Categorical-Categorical correlations
    for i in range(len(categorical_cols)):
        for j in range(i+1, len(categorical_cols)):
            col1 = categorical_cols[i]
            col2 = categorical_cols[j]
            try:
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.size > 0:
                    chi2, p, _, _ = stats.chi2_contingency(contingency)

                    is_standout = False
                    if p < standout_threshold_p_value:
                        is_standout = True

                    correlations[f"{col1} vs {col2}"] = {
                        'type': 'chi-squared',
                        'statistic': chi2,
                        'p_value': p,
                        'is_standout': is_standout # Add is_standout flag
                    }
                    correlation_interpretations[f"{col1} vs {col2}"] = f"Chi-squared statistic = {chi2:.2f}, p-value = {p:.3f}. "
                    if p < 0.05:
                        correlation_interpretations[f"{col1} vs {col2}"] += f"Significant association between '{col1}' and '{col2}'."
                    else:
                        correlation_interpretations[f"{col1} vs {col2}"] += "No significant association detected."
            except Exception as e:
                correlations[f"{col1} vs {col2}"] = {"error": str(e)}
                correlation_interpretations[f"{col1} vs {col2}"] = f"Error computing categorical-categorical correlation: {e}"

    correlation_interpretations['standout_interpretation'] = f"Standout correlations are highlighted in red. For numerical correlations, a coefficient above {standout_threshold_numerical:.1f} (absolute value) is considered strong. For other correlations, a p-value below {standout_threshold_p_value:.2f} indicates statistical significance."
    return {'correlations': correlations, 'interpretations': correlation_interpretations}

def safe_to_datetime(col):
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

@app.post("/analyze")
async def analyze_data_endpoint(file: UploadFile):
    try:
        # Read CSV file
        df = pd.read_csv(file.file, parse_dates=True)

        # Automatic datetime detection - moved before dropna for correct type detection
        df = df.apply(lambda col: safe_to_datetime(col) if col.dtype == 'object' else col)

        report = {
            "overview": {},
            "basic_stats": {},
            "graphs": {},
            "correlations": {},
            "graphs_interpretations": {},
            "correlation_interpretations": {}
        }

        # Process each column
        for col in df.columns:
            try:
                print(f"DEBUG: Processing column '{col}'")
                col_data = df[col].dropna() if df[col].dtype != object else df[col] # dropna after type detection
                report["overview"][col] = analyze_column(col_data)
                report["basic_stats"][col] = calculate_stats(df[col]) # Use original column for stats to include missing values info
                graph_results = generate_graphs(col_data) # Use non-NA data for graphs
                report["graphs"][col] = graph_results['graphs']
                report["graphs_interpretations"][col] = graph_results['interpretations']
                print(f"DEBUG: Finished processing column '{col}'")
            except Exception as e:
                print(f"DEBUG: Error processing column '{col}': {e}")
                report["overview"][col] = {"error": f"Error analyzing column: {str(e)}"}
                report["basic_stats"][col] = {"error": f"Error calculating statistics: {str(e)}"}
                report["graphs"][col] = {"error": "Error generating graphs"}
                report["graphs_interpretations"][col] = {"error": f"Error generating graph interpretations: {str(e)}"}

        # Calculate correlations
        try:
            print("DEBUG: Calculating correlations")
            corr_results = calculate_correlations(df)
            report["correlations"] = corr_results['correlations']
            report["correlation_interpretations"] = corr_results['interpretations']
            print("DEBUG: Finished calculating correlations")
        except Exception as e:
            print(f"DEBUG: Error calculating correlations: {e}")
            report["correlations"] = {"error": f"Error calculating correlations: {str(e)}"}
            report["correlation_interpretations"] = {"error": f"Error generating correlation interpretations: {str(e)}"}

        # Convert numpy types to native Python types before JSON encoding
        try:
            print("DEBUG: Converting numpy types")
            report = convert_numpy_types(report)
            print("DEBUG: Finished converting numpy types")
        except Exception as e:
            print(f"DEBUG: Error converting numpy types: {e}")
            # If conversion fails, return a simpler error response
            return JSONResponse({"error": f"Error converting data types: {str(e)}"})

        return JSONResponse(report)

    except Exception as e:
        error_message = f"Error analyzing file: {str(e)}"
        print(f"DEBUG: {error_message}")
        return JSONResponse({"error": error_message}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def main():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)