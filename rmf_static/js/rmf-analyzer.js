/**
 * RMF Analysis Tool JavaScript
 * Handles file upload, column selection, and RMF analysis
 */

// Store the currently uploaded file
let currentFile = null;

// DOM elements
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing RMF Analyzer');
    
    const fileUploadForm = document.getElementById('fileUploadForm');
    const rmfForm = document.getElementById('rmfForm');
    const csvFileInput = document.getElementById('csvFile');
    const recencyColumnSelect = document.getElementById('recencyColumn');
    const frequencyColumnSelect = document.getElementById('frequencyColumn');
    const monetaryColumnSelect = document.getElementById('monetaryColumn');
    const useAdvancedRmfCheckbox = document.getElementById('useAdvancedRmf');
    const columnSelectionSection = document.getElementById('columnSelectionSection');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const reportSection = document.getElementById('reportSection');
    const rmfContent = document.getElementById('rmfContent');
    
    // Debug log element existence
    console.log('File upload form exists:', !!fileUploadForm);
    console.log('RMF form exists:', !!rmfForm);
    console.log('CSV file input exists:', !!csvFileInput);
    
    // Handle file upload
    if (fileUploadForm) {
        fileUploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // Handle RMF analysis form submission
    if (rmfForm) {
        rmfForm.addEventListener('submit', analyzeRmf);
    }
    
    // Add global export function
    window.exportToCSV = exportToCSV;
});

/**
 * Show alert messages
 */
function showAlert(message, type = "info") {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Find a place to show the alert
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    } else {
        alert(message);
    }
}

/**
 * Handle file upload form submission
 */
function handleFileUpload(event) {
    event.preventDefault();
    
    const csvFileInput = document.getElementById('csvFile');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const columnSelectionSection = document.getElementById('columnSelectionSection');
    
    const file = csvFileInput.files[0];
    if (!file) {
        showAlert('Please select a file to upload', 'warning');
        return;
    }
    
    // Check if file is a CSV
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please upload a CSV file', 'warning');
        return;
    }
    
    console.log('File selected:', file.name);
    
    // Store the current file
    currentFile = file;
    
    // Show loading indicator
    loadingIndicator.style.display = 'block';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Send to server to get headers
    fetch('/api/headers', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Headers response status:', response.status);
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Error processing file');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Headers received:', data);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Populate column selects
        if (data.headers && data.headers.length > 0) {
            populateColumnSelects(data.headers);
            
            // Show column selection section
            columnSelectionSection.style.display = 'block';
            
            showAlert(`File loaded successfully. ${data.headers.length} columns found.`, 'success');
        } else {
            showAlert('No headers found in the file', 'danger');
        }
    })
    .catch(error => {
        console.error('Error processing file:', error);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        showAlert(`Error processing file: ${error.message}`, 'danger');
    });
}

/**
 * Populate column selects with headers from CSV file
 */
function populateColumnSelects(headers) {
    const recencyColumnSelect = document.getElementById('recencyColumn');
    const frequencyColumnSelect = document.getElementById('frequencyColumn');
    const monetaryColumnSelect = document.getElementById('monetaryColumn');
    
    // Clear existing options (except the first one)
    [recencyColumnSelect, frequencyColumnSelect, monetaryColumnSelect].forEach(select => {
        const firstOption = select.options[0];
        select.innerHTML = '';
        select.appendChild(firstOption);
    });
    
    // Add header options to each select
    headers.forEach(header => {
        [recencyColumnSelect, frequencyColumnSelect, monetaryColumnSelect].forEach(select => {
            const option = document.createElement('option');
            option.value = header;
            option.textContent = header;
            select.appendChild(option);
        });
    });
    
    // Try to auto-select columns based on common names
    headers.forEach(header => {
        const headerLower = header.toLowerCase();
        
        // Auto-select recency column (date-related)
        if (headerLower.includes('date') || 
            headerLower.includes('time') || 
            headerLower.includes('recency') || 
            headerLower.includes('last')) {
            console.log('Auto-selecting recency column:', header);
            recencyColumnSelect.value = header;
        }
        
        // Auto-select frequency column
        if (headerLower.includes('frequency') || 
            headerLower.includes('count') || 
            headerLower.includes('visits') || 
            headerLower.includes('orders')) {
            console.log('Auto-selecting frequency column:', header);
            frequencyColumnSelect.value = header;
        }
        
        // Auto-select monetary column
        if (headerLower.includes('monetary') || 
            headerLower.includes('amount') || 
            headerLower.includes('value') || 
            headerLower.includes('revenue') || 
            headerLower.includes('sales') || 
            headerLower.includes('spend')) {
            console.log('Auto-selecting monetary column:', header);
            monetaryColumnSelect.value = header;
        }
    });
}

/**
 * Handle RMF analysis form submission
 */
function analyzeRmf(event) {
    event.preventDefault();
    
    const recencyColumnSelect = document.getElementById('recencyColumn');
    const frequencyColumnSelect = document.getElementById('frequencyColumn');
    const monetaryColumnSelect = document.getElementById('monetaryColumn');
    const useAdvancedRmfCheckbox = document.getElementById('useAdvancedRmf');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const reportSection = document.getElementById('reportSection');
    
    if (!currentFile) {
        showAlert('Please upload a file first', 'warning');
        return;
    }
    
    const recencyCol = recencyColumnSelect.value;
    const frequencyCol = frequencyColumnSelect.value;
    const monetaryCol = monetaryColumnSelect.value;
    const useAdvancedRmf = useAdvancedRmfCheckbox.checked;
    
    if (!recencyCol || !frequencyCol || !monetaryCol) {
        showAlert('Please select all required columns for RMF analysis', 'warning');
        return;
    }
    
    // Show loading indicator
    loadingIndicator.style.display = 'block';
    
    // Hide results section while loading
    reportSection.style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('recency_col', recencyCol);
    formData.append('frequency_col', frequencyCol);
    formData.append('monetary_col', monetaryCol);
    formData.append('use_advanced_rmf', useAdvancedRmf);
    
    console.log('Sending RMF analysis request with columns:', {
        recency: recencyCol,
        frequency: frequencyCol,
        monetary: monetaryCol,
        advanced: useAdvancedRmf
    });
    
    // Send to server for RMF analysis
    fetch('/api/analyze-rmf', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('RMF analysis response status:', response.status);
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Error performing RMF analysis');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('RMF analysis results:', data);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Render RMF results
        renderRmfResults(data);
        
        // Show results section
        reportSection.style.display = 'block';
        
        // Scroll to results
        reportSection.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error analyzing data:', error);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        showAlert(`Error performing RMF analysis: ${error.message}`, 'danger');
    });
}

/**
 * Render RMF analysis results
 */
function renderRmfResults(rmfData) {
    const rmfContent = document.getElementById('rmfContent');
    
    if (!rmfContent) {
        console.error("Could not find element with ID 'rmfContent'");
        return;
    }
    
    if (rmfData.error) {
        rmfContent.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error calculating RMF:</strong> ${rmfData.error}
            </div>
        `;
        return;
    }
    
    if (rmfData.message) {
        rmfContent.innerHTML = `
            <div class="alert alert-info">
                ${rmfData.message}
            </div>
        `;
        return;
    }
    
    // Create HTML content
    let html = '';
    
    // Method description
    if (rmfData.method) {
        html += `<div class="alert alert-info">
            <strong>Analysis Method:</strong> ${rmfData.method}
            ${rmfData.has_clustering ? 
                '<div class="mt-2"><span class="badge bg-primary">Advanced</span> Using K-means clustering for monetary values to identify natural spending patterns.</div>' 
                : ''}
        </div>`;
    }
    
    // Columns used
    if (rmfData.columns_used) {
        html += `<div class="alert alert-secondary">
            <strong>Columns Used:</strong>
            <ul class="mb-0">
                <li>Recency: ${rmfData.columns_used.recency}</li>
                <li>Frequency: ${rmfData.columns_used.frequency}</li>
                <li>Monetary: ${rmfData.columns_used.monetary}</li>
            </ul>
        </div>`;
    }
    
    // Segment summary
    if (rmfData.segment_summary) {
        html += '<div class="segment-summary mb-4"><h4>Customer Segments</h4>';
        
        // Create segment list
        html += '<ul class="list-group">';
        for (const [segment, count] of Object.entries(rmfData.segment_summary)) {
            // Get segment class
            let segmentClass = '';
            if (segment.includes('Champions')) segmentClass = 'segment-champions';
            else if (segment.includes('Loyal')) segmentClass = 'segment-loyal';
            else if (segment.includes('Potential')) segmentClass = 'segment-potential';
            else if (segment.includes('Risk')) segmentClass = 'segment-risk';
            else if (segment.includes('Hibernating')) segmentClass = 'segment-hibernating';
            else if (segment.includes('Lost')) segmentClass = 'segment-lost';
            
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <span class="segment-badge ${segmentClass}">${segment}</span>
                    <span class="badge bg-primary rounded-pill">${count}</span>
                </li>
            `;
        }
        html += '</ul></div>';
    }
    
    // Interpretation guide
    if (rmfData.interpretation) {
        html += `<div class="mb-4">
            <h4>Interpretation Guide</h4>
            <div class="card">
                <div class="card-body">
                    <p><strong>R Score:</strong> ${rmfData.interpretation.r_score}</p>
                    <p><strong>F Score:</strong> ${rmfData.interpretation.f_score}</p>
                    <p><strong>M Score:</strong> ${rmfData.interpretation.m_score}</p>
                    <p><strong>Segments:</strong> ${rmfData.interpretation.segments}</p>
                    ${rmfData.interpretation.clustering ? `<p><strong>Clustering:</strong> ${rmfData.interpretation.clustering}</p>` : ''}
                </div>
            </div>
        </div>`;
    }
    
    // Data table
    if (rmfData.data && rmfData.data.length > 0) {
        html += `<div class="mb-4">
            <h4>RMF Analysis Results</h4>
            <div class="table-responsive">
                <table class="table table-striped table-bordered">
                    <thead class="table-dark">
                        <tr>`;
        
        // Generate table headers based on first data row
        const firstRow = rmfData.data[0];
        for (const column in firstRow) {
            html += `<th>${column}</th>`;
        }
        html += `</tr>
                    </thead>
                    <tbody>`;
        
        // Generate table rows
        rmfData.data.forEach(row => {
            html += '<tr>';
            for (const column in firstRow) {
                html += `<td>${row[column] !== undefined ? row[column] : ''}</td>`;
            }
            html += '</tr>';
        });
        
        html += `</tbody>
                </table>
            </div>
        </div>`;
        
        // Add export buttons
        html += `<div class="mb-4">
            <button class="btn btn-primary" onclick="exportToCSV()">Export to CSV</button>
        </div>`;
    }
    
    // Set the content
    rmfContent.innerHTML = html;
}

/**
 * Export RMF results to CSV file
 */
function exportToCSV() {
    const rmfContent = document.getElementById('rmfContent');
    const table = rmfContent.querySelector('table');
    
    if (!table) {
        showAlert('No data to export', 'warning');
        return;
    }
    
    try {
        // Get headers
        const headers = [];
        const headerCells = table.querySelectorAll('thead th');
        headerCells.forEach(cell => {
            headers.push(cell.textContent);
        });
        
        // Get data rows
        const rows = [];
        const dataCells = table.querySelectorAll('tbody tr');
        dataCells.forEach(row => {
            const rowData = [];
            row.querySelectorAll('td').forEach(cell => {
                // Handle CSV special characters (quote cells with commas)
                const value = cell.textContent;
                if (value.includes(',')) {
                    rowData.push(`"${value}"`);
                } else {
                    rowData.push(value);
                }
            });
            rows.push(rowData.join(','));
        });
        
        // Combine into CSV content
        const csvContent = [headers.join(','), ...rows].join('\n');
        
        // Create a blob and download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'rmf_analysis_results.csv');
        link.style.display = 'none';
        
        // Add to document, click, and remove
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showAlert('CSV file exported successfully', 'success');
    } catch (error) {
        console.error('Error exporting CSV:', error);
        showAlert('Error exporting to CSV: ' + error.message, 'danger');
    }
} 