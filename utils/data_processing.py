import numpy as np
import pandas as pd
import scipy.stats as stats
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

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

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def safe_to_datetime(col):
    """Attempt to convert a Series to datetime safely."""
    try:
        converted = pd.to_datetime(col, errors='coerce')
        if converted.notna().sum() > 0:
            print(f"DEBUG: Converted column to datetime with {converted.notna().sum()} non-NaT values.")
            return converted
        else:
            print("DEBUG: Conversion resulted in all NaT, returning original column.")
            return col
    except Exception as e:
        print(f"DEBUG: safe_to_datetime conversion error: {e}")
        return col

def check_normality(data: pd.Series) -> tuple:
    """Check if data follows normal distribution."""
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