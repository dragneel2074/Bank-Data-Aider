# main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder  # DEBUG: imported jsonable_encoder to convert numpy types to python types
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
        return False, None
    try:
        if len(data) > 5000:
            stat, p = stats.kstest(data, 'norm')
        else:
            stat, p = stats.shapiro(data)
        return (p > 0.05), p
    except:
        return False, None

def convert_numpy_types(obj):
    """Recursively convert numpy data types (like np.bool_, np.int64, etc.) to native Python types."""
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
        is_normal, p = check_normality(col_data)
        
        analysis.update({
            'is_normal': is_normal,
            'normality_test': 'Shapiro-Wilk' if len(col_data) < 5000 else 'Kolmogorov-Smirnov',
            'p_value': p,
            'recommended_tests': {
                'correlation': 'pearson' if is_normal else 'spearman',
                'group_comparison': 't-test' if is_normal else 'mannwhitneyu',
                'multiple_groups': 'anova' if is_normal else 'kruskal'
            }
        })
        
        if not is_normal:
            analysis['transform_recommendations'] = ['log', 'boxcox', 'quantile']
            
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        # Handle datetime data
        analysis['time_analysis'] = {
            'seasonality': True,
            'trend_analysis': True
        }
        
    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        # Handle categorical data
        analysis.update({
            'unique_values': len(col_data.unique()),
            'top_category': col_data.value_counts().index[0] if not col_data.empty else None
        })
    
    return analysis

def calculate_stats(col_data: pd.Series) -> Dict[str, Any]:
    stats = {}
    
    if pd.api.types.is_numeric_dtype(col_data):
        stats = {
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
        }
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        stats = {
            'min_date': col_data.min().isoformat(),
            'max_date': col_data.max().isoformat(),
            'unique_days': len(col_data.dt.normalize().unique()),
            'missing_values': col_data.isna().sum()
        }
    else:
        stats = {
            'unique_count': col_data.nunique(),
            'top_value': col_data.value_counts().index[0] if not col_data.empty else None,
            'top_value_count': col_data.value_counts().iloc[0] if not col_data.empty else None,
            'missing_values': col_data.isna().sum()
        }
    
    return stats

def generate_graphs(col_data: pd.Series) -> Dict[str, str]:
    graphs = {}
    print("DEBUG: generate_graphs called for column with dtype:", col_data.dtype, "length:", len(col_data))

    if col_data.empty:
        print("DEBUG: Empty col_data; no graphs will be generated.")
        graphs['message'] = "No data available to plot"
        return graphs

    if pd.api.types.is_numeric_dtype(col_data):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(col_data, kde=True, ax=ax)
            graphs['histogram'] = fig_to_base64(fig)
            print("DEBUG: Histogram generated successfully")
        except Exception as e:
            print("DEBUG: Error generating histogram:", e)
            graphs['histogram'] = "Error generating histogram"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=col_data, ax=ax)
            graphs['boxplot'] = fig_to_base64(fig)
            print("DEBUG: Boxplot generated successfully")
        except Exception as e:
            print("DEBUG: Error generating boxplot:", e)
            graphs['boxplot'] = "Error generating boxplot"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(col_data.dropna(), plot=ax)
            graphs['qq_plot'] = fig_to_base64(fig)
            print("DEBUG: Q-Q plot generated successfully")
        except Exception as e:
            print("DEBUG: Error generating Q-Q plot:", e)
            graphs['qq_plot'] = "Error generating Q-Q plot"
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            col_data.groupby(col_data.dt.floor('d')).count().plot(ax=ax)
            ax.set_title('Transaction Density Over Time')
            graphs['time_density'] = fig_to_base64(fig)
            print("DEBUG: Time density plot generated successfully")
        except Exception as e:
            print("DEBUG: Error generating time density plot:", e)
            graphs['time_density'] = "Error generating time density plot"
        # try:
        #     ts_series = col_data.dropna().resample('D').count()
        #     print("DEBUG: ts_series length for seasonal decomposition:", len(ts_series))
        #     if len(ts_series) >= 60:  # require at least two periods (2*30)
        #         result = seasonal_decompose(ts_series, period=30)
        #         fig = result.plot()
        #         graphs['seasonality'] = fig_to_base64(fig)
        #         print("DEBUG: Seasonal decomposition plot generated successfully")
        #     else:
        #         print("DEBUG: Not enough data for seasonal decomposition; length:", len(ts_series))
        #         graphs['seasonality'] = "Insufficient data for seasonal decomposition"
        # except Exception as e:
        #     print("DEBUG: Error in seasonal decomposition:", e)
        #     graphs['seasonality'] = "Error generating seasonal decomposition plot"
    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            col_data.value_counts().head(10).plot(kind='bar', ax=ax)
            graphs['category_distribution'] = fig_to_base64(fig)
            print("DEBUG: Category distribution plot generated successfully")
        except Exception as e:
            print("DEBUG: Error generating category distribution plot:", e)
            graphs['category_distribution'] = "Error generating category distribution plot"
    return graphs

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    correlations = {}
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Numerical-Numerical correlations
    if len(numerical_cols) > 1:
        try:
            corr_matrix = df[numerical_cols].corr(method='spearman')
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    try:
                        test_type = 'spearman'
                        if check_normality(df[col1])[0] and check_normality(df[col2])[0]:
                            test_type = 'pearson'
                        result = stats.spearmanr(df[col1], df[col2])
                        correlations[f"{col1} vs {col2}"] = {
                            'type': test_type,
                            'coefficient': corr_matrix.iloc[i, j],
                            'p_value': result.pvalue
                        }
                        print(f"DEBUG: Numerical correlation computed for {col1} vs {col2}")
                    except Exception as e:
                        print(f"DEBUG: Error computing numerical correlation for {col1} vs {col2}: {e}")
                        correlations[f"{col1} vs {col2}"] = {"error": str(e)}
        except Exception as e:
            print("DEBUG: Error computing numerical correlation matrix:", e)
    
    # Numerical-Categorical correlations
    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            if df[cat_col].nunique() > 1:
                groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    try:
                        _, normal = check_normality(df[num_col])
                        if normal:
                            stat, p = stats.f_oneway(*groups)
                            test_type = 'anova'
                        else:
                            stat, p = stats.kruskal(*groups)
                            test_type = 'kruskal'
                        correlations[f"{num_col} vs {cat_col}"] = {
                            'type': test_type,
                            'statistic': stat,
                            'p_value': p
                        }
                        print(f"DEBUG: Numerical-Categorical correlation computed for {num_col} vs {cat_col}")
                    except Exception as e:
                        print(f"DEBUG: Error computing numerical-categorical correlation for {num_col} vs {cat_col}: {e}")
                        correlations[f"{num_col} vs {cat_col}"] = {"error": str(e)}
    
    # Categorical-Categorical correlations
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
                        'p_value': p
                    }
                    print(f"DEBUG: Categorical-Categorical correlation computed for {col1} vs {col2}")
            except Exception as e:
                print(f"DEBUG: Error computing categorical-categorical correlation for {col1} vs {col2}: {e}")
                correlations[f"{col1} vs {col2}"] = {"error": str(e)}
    
    return correlations

@app.post("/analyze")
async def analyze_data(file: UploadFile):
    try:
        # Read CSV file
        df = pd.read_csv(file.file, parse_dates=True)
        
        # Automatic datetime detection
        df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
                      if col.dtype == 'object' else col)
        
        report = {
            "overview": {},
            "basic_stats": {},
            "graphs": {},
            "correlations": {}
        }
        
        # Process each column
        for col in df.columns:
            col_data = df[col].dropna() if df[col].dtype != object else df[col]
            
            report["overview"][col] = analyze_column(col_data)
            report["basic_stats"][col] = calculate_stats(col_data)
            report["graphs"][col] = generate_graphs(col_data)
        
        # Calculate correlations
        report["correlations"] = calculate_correlations(df)
        
        # Convert numpy types to native Python types before JSON encoding
        report = convert_numpy_types(report)  # DEBUG: custom conversion of numpy types to native Python types
        print("DEBUG: report after custom conversion:", report)  # DEBUG: print the converted report
        return JSONResponse(report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def main():
    # DEBUG: Serving the UI from static/html/index.html
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)