from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from services.analysis_service import (
    analyze_column,
    calculate_stats,
    generate_graphs,
    generate_pairwise_graphs,
    calculate_correlations,
    calculate_rmf,
    calculate_rmf_kmeans
)
from utils.data_processing import convert_numpy_types
import io

# Create router
analysis_router = APIRouter()

@analysis_router.post("/headers")
async def get_csv_headers(file: UploadFile = File(...)):
    """Get headers from uploaded CSV file."""
    try:
        print(f"Processing file: {file.filename}")  # Debug print
        contents = await file.read()
        
        # Create a temporary file-like object in memory
        csv_file = io.StringIO(contents.decode())
        
        # Read CSV headers
        df = pd.read_csv(csv_file, parse_dates=True)
        headers = df.columns.tolist()
        
        print(f"Found headers: {headers}")  # Debug print
        return {"headers": headers}
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")  # Debug print
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

# Pydantic models for request validation
class ColumnData(BaseModel):
    column_data: List[Any]
    column_name: str

class PairwiseData(BaseModel):
    data: List[Dict[str, Any]]
    column1_name: str
    column2_name: str

class CorrelationData(BaseModel):
    data: List[Dict[str, Any]]

@analysis_router.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    headers_to_process: List[str] = Form(...),
    recency_col: Optional[str] = Form(None),
    frequency_col: Optional[str] = Form(None), 
    monetary_col: Optional[str] = Form(None),
    use_advanced_rmf: Optional[str] = Form("false"),
    run_rmf_analysis: Optional[str] = Form(None),
    datetime_columns: Optional[List[str]] = Form([]),
    categorical_columns: Optional[List[str]] = Form([])
):
    """Analyze the uploaded CSV file data."""
    try:
        print(f"Analyzing file: {file.filename}")  # Debug print
        print(f"Headers to process: {headers_to_process}")  # Debug print
        print(f"Run RMF analysis: {run_rmf_analysis}")  # Debug print
        print(f"RMF columns - Recency: {recency_col}, Frequency: {frequency_col}, Monetary: {monetary_col}")  # Debug print
        print(f"Using advanced RMF: {use_advanced_rmf}")  # Debug print
        print(f"DateTime columns: {datetime_columns}")  # Debug print
        print(f"Categorical columns: {categorical_columns}")  # Debug print
        
        # Filter out invalid headers (like 'on' or empty strings)
        filtered_headers = [h for h in headers_to_process if h and h != 'on']
        
        if not filtered_headers:
            return JSONResponse({
                "error": "No valid headers provided for analysis."
            }, status_code=400)
        
        contents = await file.read()
        csv_file = io.StringIO(contents.decode())
        
        # Parse with appropriate date format detection for datetime columns
        parse_dates = datetime_columns if datetime_columns else True
        df = pd.read_csv(csv_file, parse_dates=parse_dates)
        
        # Convert categorical columns correctly
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    print(f"Converted {col} to categorical type")  # Debug print
        
        # Check that all headers exist in the dataframe
        missing_headers = [h for h in filtered_headers if h not in df.columns]
        if missing_headers:
            return JSONResponse({
                "error": f"The following headers do not exist in the uploaded file: {', '.join(missing_headers)}"
            }, status_code=400)
        
        df_processed = df[filtered_headers]
        
        # Process each column
        report = {
            "summary": {
                "file_name": file.filename,
                "total_rows": len(df),
                "total_columns": len(filtered_headers),
                "columns_analyzed": filtered_headers,
                "datetime_columns": datetime_columns,
                "categorical_columns": categorical_columns
            },
            "distributions": {},
            "correlations": {},
            "graphs": {},
            "pairwise_graphs": {},
            "interpretations": {},
            "rmf_results": None
        }
        
        for col in df_processed.columns:
            try:
                print(f"Processing column: {col}")  # Debug print
                print(f"Column data type: {df_processed[col].dtype}")  # Debug print
                
                # For datetime columns, use the appropriate handling
                if col in datetime_columns:
                    print(f"Processing {col} as DateTime column")  # Debug print
                    # Ensure it's properly converted to datetime
                    if not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                        print(f"Converted {col} to datetime type")  # Debug print
                
                # For categorical columns, use the appropriate handling
                elif col in categorical_columns:
                    print(f"Processing {col} as Categorical column")  # Debug print
                    # Ensure it's properly converted to categorical
                    if not pd.api.types.is_categorical_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].astype('category')
                        print(f"Converted {col} to categorical type")  # Debug print
                
                col_data = df_processed[col].dropna() if df_processed[col].dtype != object else df_processed[col]
                report["distributions"][col] = analyze_column(col_data)
                report["distributions"][col].update(calculate_stats(df_processed[col]))
                graph_results = generate_graphs(col_data)
                report["graphs"][col] = graph_results['graphs']
                report["interpretations"][col] = graph_results['interpretations']
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")  # Debug print
                report["distributions"][col] = {"error": f"Error analyzing column: {str(e)}"}
                report["graphs"][col] = {"error": "Error generating graphs"}
                report["interpretations"][col] = {"error": f"Error generating interpretations: {str(e)}"}
        
        # Calculate correlations
        try:
            print("Calculating correlations")  # Debug print
            corr_results = calculate_correlations(df_processed)
            report["correlations"] = corr_results['correlations']
            report["interpretations"]["correlations"] = corr_results['interpretations']
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")  # Debug print
            report["correlations"] = {"error": f"Error calculating correlations: {str(e)}"}
            report["interpretations"]["correlations"] = {"error": f"Error generating correlation interpretations: {str(e)}"}
        
        # Generate pairwise graphs for each pair of columns
        columns = df_processed.columns.tolist()
        if len(columns) > 1:
            print("Generating pairwise graphs")  # Debug print
            for i in range(len(columns)):
                col1 = columns[i]
                report["pairwise_graphs"][col1] = {}
                if "pairwise" not in report["interpretations"]:
                    report["interpretations"]["pairwise"] = {}
                report["interpretations"]["pairwise"][col1] = {}
                for j in range(len(columns)):
                    if i != j:
                        col2 = columns[j]
                        try:
                            print(f"Generating pairwise graph for {col1} vs {col2}")  # Debug print
                            pairwise_results = generate_pairwise_graphs(df_processed, col1, col2)
                            report["pairwise_graphs"][col1][col2] = pairwise_results['graphs']
                            report["interpretations"]["pairwise"][col1][col2] = pairwise_results['interpretations']
                        except Exception as e:
                            print(f"Error generating pairwise graph for {col1} vs {col2}: {str(e)}")  # Debug print
                            report["pairwise_graphs"][col1][col2] = {"error": f"Error generating pairwise graph: {str(e)}"}
                            report["interpretations"]["pairwise"][col1][col2] = {"error": f"Error generating interpretation: {str(e)}"}
        
        # Add RMF analysis if required columns are provided
        is_rmf_requested = run_rmf_analysis == "true"
        is_advanced_rmf = use_advanced_rmf == "true"
        
        if is_rmf_requested and recency_col and frequency_col and monetary_col:
            try:
                print(f"Calculating RMF using columns: {recency_col}, {frequency_col}, {monetary_col}")  # Debug print
                print(f"Using advanced RMF algorithm: {is_advanced_rmf}")  # Debug print
                
                # Check if columns exist in the dataframe
                missing_cols = []
                for col in [recency_col, frequency_col, monetary_col]:
                    if col not in df.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    report["rmf_results"] = {
                        "error": f"Missing columns for RMF analysis: {', '.join(missing_cols)}"
                    }
                else:
                    # Calculate RMF using either standard or advanced method
                    try:
                        if is_advanced_rmf:
                            try:
                                rmf_results = calculate_rmf_kmeans(df, recency_col, frequency_col, monetary_col)
                                rmf_method = "advanced (K-means clustering)"
                            except Exception as e:
                                print(f"Error with advanced RMF, falling back to standard: {e}")
                                rmf_results = calculate_rmf(df, recency_col, frequency_col, monetary_col)
                                rmf_method = "standard (fallback from advanced)"
                        else:
                            rmf_results = calculate_rmf(df, recency_col, frequency_col, monetary_col)
                            rmf_method = "standard"

                        # Check if rmf_results is valid
                        if rmf_results is None or len(rmf_results) == 0:
                            report["rmf_results"] = {
                                "error": "RMF calculation returned empty results"
                            }
                        else:
                            # Add original values to the results for reference
                            try:
                                common_index = df.index.intersection(rmf_results.index)
                                
                                # Rename columns in original_values to avoid duplicates
                                original_values = df.loc[common_index, [recency_col, frequency_col, monetary_col]].copy()
                                original_values.columns = [f"Original_{recency_col}", f"Original_{frequency_col}", f"Original_{monetary_col}"]
                                
                                # Concat and handle NaN values
                                rmf_with_orig = pd.concat([original_values, rmf_results], axis=1)
                                
                                # Replace NaN values with "N/A" for JSON serialization
                                # Also ensure all RMF scores are integers to prevent decimal points
                                for col in ['R_Score', 'F_Score', 'M_Score']:
                                    if col in rmf_with_orig.columns:
                                        rmf_with_orig[col] = rmf_with_orig[col].fillna(1).astype(int)
                                
                                # Convert to dict with NaN handling
                                rmf_dict = convert_numpy_types(rmf_with_orig.fillna("N/A").to_dict(orient='records'))
                                
                                # Generate segment summary
                                segment_counts = rmf_results['Segment_Description'].value_counts().to_dict()
                                
                                # Check if we have clustering information
                                has_clusters = 'M_Cluster' in rmf_results.columns
                                
                                # Prepare the RMF report section
                                report["rmf_results"] = {
                                    "data": rmf_dict,
                                    "segment_summary": segment_counts,
                                    "method": rmf_method,
                                    "has_clustering": has_clusters,
                                    "columns_used": {
                                        "recency": recency_col,
                                        "frequency": frequency_col,
                                        "monetary": monetary_col
                                    },
                                    "interpretation": {
                                        "r_score": "R (Recency) - Higher scores (4-5) indicate recent activity, lower scores (1-2) indicate customers who haven't engaged recently.",
                                        "f_score": "F (Frequency) - Higher scores indicate customers who engage frequently, lower scores indicate infrequent interaction.",
                                        "m_score": "M (Monetary) - Higher scores indicate customers with high monetary value, lower scores indicate lower value customers.",
                                        "segments": "Customers are segmented based on their RMF scores into categories like Champions, Loyal Customers, At Risk, etc."
                                    }
                                }
                                
                                # Add clustering explanation if available
                                if has_clusters:
                                    report["rmf_results"]["interpretation"]["clustering"] = (
                                        "M_Cluster shows the monetary value cluster assigned by K-means clustering. "
                                        "Customers in the same cluster have similar spending patterns after log transformation and scaling."
                                    )
                            except Exception as e:
                                print(f"Error preparing RMF report: {e}")
                                report["rmf_results"] = {
                                    "error": f"Error preparing RMF results: {e}",
                                    "message": "RMF scores were calculated, but there was an error formatting the results."
                                }
                    except Exception as e:
                        print(f"Error calculating RMF: {str(e)}")
                        report["rmf_results"] = {"error": f"Error calculating RMF: {str(e)}"}
            except Exception as e:
                print(f"Error in RMF analysis section: {str(e)}")
                report["rmf_results"] = {"error": f"Error in RMF analysis: {str(e)}"}
        elif is_rmf_requested:
            report["rmf_results"] = {
                "message": "RMF analysis not performed. Please provide Recency, Frequency, and Monetary columns."
            }
        
        # Convert numpy types to Python native types
        report = convert_numpy_types(report)
        
        return report
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@analysis_router.post("/analyze-column")
async def analyze_column_route(data: ColumnData):
    """Analyze a single column of data."""
    try:
        # Convert list to pandas Series
        column_data = pd.Series(data.column_data, name=data.column_name)
        
        # Perform analysis
        analysis_results = analyze_column(column_data)
        
        # Convert numpy types to Python native types
        analysis_results = convert_numpy_types(analysis_results)
        
        return {
            'success': True,
            'analysis': analysis_results
        }

    except Exception as e:
        print(f"Error in analyze_column_route: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/calculate-stats")
async def calculate_stats_route(data: ColumnData):
    """Calculate statistics for a column."""
    try:
        # Convert list to pandas Series
        column_data = pd.Series(data.column_data, name=data.column_name)
        
        # Calculate statistics
        stats_results = calculate_stats(column_data)
        
        # Convert numpy types to Python native types
        stats_results = convert_numpy_types(stats_results)
        
        return {
            'success': True,
            'statistics': stats_results
        }

    except Exception as e:
        print(f"Error in calculate_stats_route: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/generate-graphs")
async def generate_graphs_route(data: ColumnData):
    """Generate visualization graphs for a column."""
    try:
        # Convert list to pandas Series
        column_data = pd.Series(data.column_data, name=data.column_name)
        
        # Generate graphs
        graph_results = generate_graphs(column_data)
        
        return {
            'success': True,
            'graphs': graph_results['graphs'],
            'interpretations': graph_results['interpretations']
        }

    except Exception as e:
        print(f"Error in generate_graphs_route: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/generate-pairwise-graphs")
async def generate_pairwise_graphs_route(data: PairwiseData):
    """Generate pairwise visualization graphs."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data.data)
        
        # Generate pairwise graphs
        graph_results = generate_pairwise_graphs(
            df,
            data.column1_name,
            data.column2_name
        )
        
        return {
            'success': True,
            'graphs': graph_results['graphs'],
            'interpretations': graph_results['interpretations']
        }

    except Exception as e:
        print(f"Error in generate_pairwise_graphs_route: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/calculate-correlations")
async def calculate_correlations_route(data: CorrelationData):
    """Calculate correlations between variables."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data.data)
        
        # Calculate correlations
        correlation_results = calculate_correlations(df)
        
        # Convert numpy types to Python native types
        correlation_results = convert_numpy_types(correlation_results)
        
        return {
            'success': True,
            'correlations': correlation_results['correlations'],
            'interpretations': correlation_results['interpretations']
        }

    except Exception as e:
        print(f"Error in calculate_correlations_route: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e)) 