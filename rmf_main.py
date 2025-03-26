from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import pandas as pd
import os
import io
import numpy as np
import uvicorn
from services.analysis_service import calculate_rmf, calculate_rmf_kmeans
from utils.data_processing import convert_numpy_types

# Create FastAPI app
app = FastAPI(
    title="RMF Analysis System",
    description="Standalone API for performing RMF analysis on customer data",
    version="1.0.0"
)

# Mount static files
os.makedirs("rmf_static", exist_ok=True)
app.mount("/static", StaticFiles(directory="rmf_static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="rmf_templates")

# Serve the HTML interface
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/headers")
async def get_csv_headers(file: UploadFile = File(...)):
    """Get headers from uploaded CSV file."""
    try:
        print(f"Processing file: {file.filename}")
        contents = await file.read()
        
        # Create a temporary file-like object in memory
        csv_file = io.StringIO(contents.decode())
        
        # Read CSV headers
        df = pd.read_csv(csv_file, parse_dates=True)
        headers = df.columns.tolist()
        
        print(f"Found headers: {headers}")
        return {"headers": headers}
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Error reading CSV file: {str(e)}"}
        )
    
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

@app.post("/api/analyze-rmf")
async def analyze_rmf(
    file: UploadFile = File(...),
    recency_col: str = Form(...),
    frequency_col: str = Form(...),
    monetary_col: str = Form(...),
    use_advanced_rmf: bool = Form(False)
):
    """Perform RMF analysis on the uploaded file."""
    try:
        print(f"Analyzing RMF for file: {file.filename}")
        print(f"RMF columns - Recency: {recency_col}, Frequency: {frequency_col}, Monetary: {monetary_col}")
        print(f"Using advanced RMF: {use_advanced_rmf}")
        
        # Read the CSV file
        contents = await file.read()
        csv_file = io.StringIO(contents.decode())
        df = pd.read_csv(csv_file, parse_dates=True)
        
        # Check that all required columns exist in the dataframe
        missing_headers = []
        for col in [recency_col, frequency_col, monetary_col]:
            if col not in df.columns:
                missing_headers.append(col)
        
        if missing_headers:
            return JSONResponse(
                status_code=400,
                content={"error": f"The following columns do not exist in the uploaded file: {', '.join(missing_headers)}"}
            )
        
        # Perform RMF analysis
        try:
            # Choose the correct RMF calculation method
            if use_advanced_rmf:
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
            
            # Validate results
            if rmf_results is None or len(rmf_results) == 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "RMF calculation returned empty results"}
                )
            
            # Add original values to the results for reference
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
            
            # Prepare the RMF report
            rmf_report = {
                "data": rmf_dict[:200],
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
            # rmf_df = pd.DataFrame(rmf_results)
            # rmf_df.to_csv("rmf_results.csv", index=False)
            # print(f"RMF results saved to rmf_results.csv")
            try:
                # Convert the RMF results list to a DataFrame
                rmf_df = pd.DataFrame(rmf_results)
                print("DEBUG: Converted RMF results to DataFrame with shape:", rmf_df.shape)
                # Ensure the indices align (resetting if necessary)
                df_reset = df.reset_index(drop=True)
                rmf_df_reset = rmf_df.reset_index(drop=True)
                # Merge the original DataFrame with the RMF results DataFrame along the columns
                merged_df = pd.concat([df_reset, rmf_df_reset], axis=1)
                print("DEBUG: Merged RMF results with original DataFrame. Final shape:", merged_df.shape)
                # Save the merged DataFrame as a CSV file
                merged_df.to_csv("rmf_results.csv", index=False)
                print("DEBUG: RMF results saved to rmf_results.csv")
            except Exception as e:
                print("DEBUG: Failed to merge and save RMF results:", e)
            # Add clustering explanation if available
            if has_clusters:
                rmf_report["interpretation"]["clustering"] = (
                    "M_Cluster shows the monetary value cluster assigned by K-means clustering. "
                    "Customers in the same cluster have similar spending patterns after log transformation and scaling."
                )
            
            return rmf_report
            
        except Exception as e:
            print(f"Error calculating RMF: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error calculating RMF: {str(e)}"}
            )
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

if __name__ == "__main__":
    # Create the templates directory if it doesn't exist
    os.makedirs("rmf_templates", exist_ok=True)
    
    # Enable debug mode for development
    debug_mode = os.environ.get('ENVIRONMENT') == 'development'
    print(f"Starting RMF analysis server in {'debug' if debug_mode else 'production'} mode")
    uvicorn.run("rmf_main:app", host="0.0.0.0", port=8001, reload=debug_mode) 