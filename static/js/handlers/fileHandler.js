/**
 * Handles file upload and header generation
 */
export function handleFileUpload(event) {
    console.log('File upload handler called');
    
    const file = event.target.files[0];
    if (!file) {
        console.log('No file selected');
        return;
    }
    
    console.log(`File selected: ${file.name} (${file.size} bytes)`);
    
    // Clear previous results
    const resultsDiv = document.getElementById('results');
    if (resultsDiv) resultsDiv.style.display = 'none';
    
    const pairwiseSection = document.getElementById('pairwiseGraphsSection');
    if (pairwiseSection) pairwiseSection.style.display = 'none';
    
    const rmfSection = document.getElementById('rmfSection');
    if (rmfSection) rmfSection.style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show a loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    
    // Get headers from the file
    fetch('/api/analysis/headers', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Headers received:', data.headers);
        
        // Generate checkboxes for headers
        generateHeaderCheckboxes(data.headers);
        
        // Update RMF column dropdowns
        if (typeof window.updateRmfColumnOptions === 'function') {
            window.updateRmfColumnOptions(data.headers);
        }
        
        // Hide loading indicator
        if (loadingIndicator) loadingIndicator.style.display = 'none';
    })
    .catch(error => {
        console.error('Error getting headers:', error);
        alert(`Failed to process file: ${error.message}`);
        
        // Hide loading indicator
        if (loadingIndicator) loadingIndicator.style.display = 'none';
    });
}

/**
 * Generates checkboxes for selecting headers to analyze
 */
function generateHeaderCheckboxes(headers) {
    console.log('Generating header checkboxes for', headers);
    
    const checkboxContainer = document.getElementById('headerCheckboxes');
    if (!checkboxContainer) {
        console.error('Header checkbox container not found');
        return;
    }
    
    checkboxContainer.innerHTML = '';
    
    // Add a "Select All" checkbox at the top
    const selectAllDiv = document.createElement('div');
    selectAllDiv.className = 'col-12 mb-3';
    
    const selectAllCheck = document.createElement('div');
    selectAllCheck.className = 'form-check';
    
    const selectAllCheckbox = document.createElement('input');
    selectAllCheckbox.type = 'checkbox';
    selectAllCheckbox.className = 'form-check-input';
    selectAllCheckbox.id = 'selectAllHeaders';
    selectAllCheckbox.value = 'selectAll'; // Give it a distinct value
    selectAllCheckbox.checked = true;
    
    const selectAllLabel = document.createElement('label');
    selectAllLabel.className = 'form-check-label fw-bold';
    selectAllLabel.htmlFor = 'selectAllHeaders';
    selectAllLabel.textContent = 'Select/Deselect All';
    
    selectAllCheck.appendChild(selectAllCheckbox);
    selectAllCheck.appendChild(selectAllLabel);
    selectAllDiv.appendChild(selectAllCheck);
    checkboxContainer.appendChild(selectAllDiv);
    
    // Add event listener to select all checkbox
    selectAllCheckbox.addEventListener('change', function() {
        const allCheckboxes = document.querySelectorAll('#headerCheckboxes input[type="checkbox"]:not(#selectAllHeaders)');
        allCheckboxes.forEach(checkbox => {
            checkbox.checked = selectAllCheckbox.checked;
        });
    });
    
    // Create checkbox for each header in 3-column layout
    headers.forEach((header, index) => {
        const checkDiv = document.createElement('div');
        checkDiv.className = 'col-md-4 col-sm-6 header-checkbox-item';
        
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'form-check';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'form-check-input';
        checkbox.value = header; // Explicitly set the value to the header name
        checkbox.id = `header_${header.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '')}`;
        checkbox.checked = true; // Default all to checked
        
        const label = document.createElement('label');
        label.className = 'form-check-label';
        label.htmlFor = checkbox.id;
        label.textContent = header;
        
        // Add event listener to individual checkboxes to update select all checkbox
        checkbox.addEventListener('change', function() {
            const allChecked = Array.from(
                document.querySelectorAll('#headerCheckboxes input[type="checkbox"]:not(#selectAllHeaders)')
            ).every(cb => cb.checked);
            
            selectAllCheckbox.checked = allChecked;
        });
        
        checkboxDiv.appendChild(checkbox);
        checkboxDiv.appendChild(label);
        checkDiv.appendChild(checkboxDiv);
        checkboxContainer.appendChild(checkDiv);
    });
    
    // Show header selection section
    const headerSection = document.getElementById('headerSelectionSection');
    if (headerSection) {
        headerSection.style.display = 'block';
    }
    
    // Update RMF column dropdown options
    if (typeof window.updateRmfColumnOptions === 'function') {
        window.updateRmfColumnOptions(headers);
    }
}

/**
 * Handler for the analyze button
 * Sends selected headers to the backend for analysis
 */
export function analyzeData(recencyCol, frequencyCol, monetaryCol) {
    console.log('Analyze data function called');
    
    // Get the checkboxes and file input
    const checkboxes = document.querySelectorAll('#headerCheckboxes input[type="checkbox"]:checked:not(#selectAllHeaders)');
    const fileInput = document.getElementById('csvFile');
    const analyzeBtn = document.getElementById('analyzeButton');
    const progressBar = document.getElementById('analyzeProgress');
    const progressContainer = document.getElementById('progressContainer');
    
    // Validate file selection
    if (!fileInput.files[0]) {
        alert('Please select a file first');
        return;
    }
    
    // Validate header selection
    if (checkboxes.length === 0) {
        alert('Please select at least one header to analyze');
        return;
    }
    
    // Get selected headers - make sure we only get valid column names
    const selectedHeaders = [];
    checkboxes.forEach(cb => {
        // Only add the value if it's not 'on' (default checkbox value when not explicitly set)
        if (cb.value && cb.value !== 'on') {
            selectedHeaders.push(cb.value);
        }
    });
    
    console.log('Selected headers:', selectedHeaders);
    console.log('RMF columns:', { recencyCol, frequencyCol, monetaryCol });
    
    // Ensure we have headers to analyze
    if (selectedHeaders.length === 0) {
        alert('No valid headers selected. Please select columns to analyze.');
        return;
    }
    
    // Create form data to send to the backend
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Add each selected header to form data
    selectedHeaders.forEach(header => {
        formData.append('headers_to_process', header);
    });
    
    // Add RMF columns if provided
    if (recencyCol) formData.append('recency_col', recencyCol);
    if (frequencyCol) formData.append('frequency_col', frequencyCol);
    if (monetaryCol) formData.append('monetary_col', monetaryCol);
    
    // Show progress indicator and disable button
    analyzeBtn.disabled = true;
    if (progressContainer) progressContainer.style.display = 'block';
    if (progressBar) progressBar.style.width = '25%';
    
    // Send data to backend
    fetch('/api/analysis/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (progressBar) progressBar.style.width = '75%';
        
        return response.json().then(data => {
            if (!response.ok) {
                // Use the error message from the server if available
                throw new Error(data.error || data.detail || `Server error: ${response.status}`);
            }
            return data;
        });
    })
    .then(data => {
        if (progressBar) progressBar.style.width = '100%';
        
        console.log('Analysis complete, rendering report', data);
        window.renderReport(data);
        
        // Re-enable button and hide progress after short delay
        setTimeout(() => {
            analyzeBtn.disabled = false;
            if (progressContainer) progressContainer.style.display = 'none';
            if (progressBar) progressBar.style.width = '0%';
        }, 500);
    })
    .catch(error => {
        console.error('Error during analysis:', error);
        alert(`Analysis failed: ${error.message}`);
        
        // Re-enable button and hide progress
        analyzeBtn.disabled = false;
        if (progressContainer) progressContainer.style.display = 'none';
        if (progressBar) progressBar.style.width = '0%';
    });
} 