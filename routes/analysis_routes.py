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
    calculate_rmf
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
    monetary_col: Optional[str] = Form(None)
):
    """Analyze the uploaded CSV file data."""
    try:
        print(f"Analyzing file: {file.filename}")  # Debug print
        print(f"Headers to process: {headers_to_process}")  # Debug print
        print(f"RMF columns - Recency: {recency_col}, Frequency: {frequency_col}, Monetary: {monetary_col}")  # Debug print
        
        contents = await file.read()
        csv_file = io.StringIO(contents.decode())
        df = pd.read_csv(csv_file, parse_dates=True)
        
        if headers_to_process:
            df_processed = df[headers_to_process]
        else:
            df_processed = df
            
        # Process each column
        report = {
            "headers": df_processed.columns.tolist(),
            "overview": {},
            "basic_stats": {},
            "graphs": {},
            "correlations": {},
            "graphs_interpretations": {},
            "correlation_interpretations": {},
            "pairwise_graphs": {},
            "pairwise_graphs_interpretations": {},
            "rmf_analysis": {}
        }
        
        for col in df_processed.columns:
            try:
                print(f"Processing column: {col}")  # Debug print
                col_data = df_processed[col].dropna() if df_processed[col].dtype != object else df_processed[col]
                report["overview"][col] = analyze_column(col_data)
                report["basic_stats"][col] = calculate_stats(df_processed[col])
                graph_results = generate_graphs(col_data)
                report["graphs"][col] = graph_results['graphs']
                report["graphs_interpretations"][col] = graph_results['interpretations']
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")  # Debug print
                report["overview"][col] = {"error": f"Error analyzing column: {str(e)}"}
                report["basic_stats"][col] = {"error": f"Error calculating statistics: {str(e)}"}
                report["graphs"][col] = {"error": "Error generating graphs"}
                report["graphs_interpretations"][col] = {"error": f"Error generating graph interpretations: {str(e)}"}
        
        # Calculate correlations
        try:
            print("Calculating correlations")  # Debug print
            corr_results = calculate_correlations(df_processed)
            report["correlations"] = corr_results['correlations']
            report["correlation_interpretations"] = corr_results['interpretations']
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")  # Debug print
            report["correlations"] = {"error": f"Error calculating correlations: {str(e)}"}
            report["correlation_interpretations"] = {"error": f"Error generating correlation interpretations: {str(e)}"}
        
        # Generate pairwise graphs for each pair of columns
        columns = df_processed.columns.tolist()
        if len(columns) > 1:
            print("Generating pairwise graphs")  # Debug print
            for i in range(len(columns)):
                col1 = columns[i]
                report["pairwise_graphs"][col1] = {}
                report["pairwise_graphs_interpretations"][col1] = {}
                for j in range(len(columns)):
                    if i != j:
                        col2 = columns[j]
                        try:
                            print(f"Generating pairwise graph for {col1} vs {col2}")  # Debug print
                            pairwise_results = generate_pairwise_graphs(df_processed, col1, col2)
                            report["pairwise_graphs"][col1][col2] = pairwise_results['graphs']
                            report["pairwise_graphs_interpretations"][col1][col2] = pairwise_results['interpretations']
                        except Exception as e:
                            print(f"Error generating pairwise graph for {col1} vs {col2}: {str(e)}")  # Debug print
                            report["pairwise_graphs"][col1][col2] = {"error": f"Error generating pairwise graph: {str(e)}"}
                            report["pairwise_graphs_interpretations"][col1][col2] = {"error": f"Error generating interpretation: {str(e)}"}
        
        # Add RMF analysis if required columns are provided
        if recency_col and frequency_col and monetary_col:
            try:
                print(f"Calculating RMF using columns: {recency_col}, {frequency_col}, {monetary_col}")  # Debug print
                
                # Check if columns exist in the dataframe
                missing_cols = []
                for col in [recency_col, frequency_col, monetary_col]:
                    if col not in df.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    report["rmf_analysis"] = {
                        "error": f"Missing columns for RMF analysis: {', '.join(missing_cols)}"
                    }
                else:
                    # Calculate RMF
                    rmf_results = calculate_rmf(df, recency_col, frequency_col, monetary_col)
                    
                    # Add original values to the results for reference
                    rmf_results_with_orig = df[[recency_col, frequency_col, monetary_col]].join(rmf_results)
                    
                    # Generate segment summary
                    segment_counts = rmf_results['Segment_Description'].value_counts().to_dict()
                    
                    report["rmf_analysis"] = {
                        "data": convert_numpy_types(rmf_results_with_orig.to_dict(orient='records')),
                        "segment_summary": segment_counts,
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
            except Exception as e:
                print(f"Error calculating RMF: {str(e)}")  # Debug print
                report["rmf_analysis"] = {"error": f"Error calculating RMF: {str(e)}"}
        else:
            report["rmf_analysis"] = {
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