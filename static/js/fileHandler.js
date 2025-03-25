// Store the currently uploaded file
export let currentFile = null;

// Import renderers
import { renderSummaryStats, renderDistributions, renderCorrelations, renderRmf } from './renderers/statsRenderer.js';
import { renderGraphs, renderPairwiseGraphs } from './renderers/graphRenderer.js';

/**
 * Handles file upload and processes CSV headers
 */
export function handleFileUpload(event) {
    console.log('File upload triggered');
    const file = event.target.files[0];
    if (!file) {
        console.warn('No file selected');
        return;
    }
    
    console.log('File selected:', file.name);
    currentFile = file;
    
    // Check if file is a CSV
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please upload a CSV file.', 'warning');
        return;
    }
    
    // Show loading animation
    document.getElementById('loadingIndicator').style.display = 'block';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('Sending file to get headers');
    
    // Send to server to get headers
    fetch('/api/analysis/headers', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Received headers response:', response.status, response.statusText);
        if (!response.ok) {
            return response.json().then(data => {
                console.error('Error in headers response:', data);
                throw new Error(data.error || 'Error processing file');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Headers received:', data);
        
        // Hide loading animation
        document.getElementById('loadingIndicator').style.display = 'none';
        
        // Generate checkboxes for headers
        if (data.headers && data.headers.length > 0) {
            generateHeaderCheckboxes(data.headers);
            
            // Show analyze button
            const analyzeBtn = document.getElementById('analyzeButton');
            if (analyzeBtn) {
                analyzeBtn.style.display = 'block';
            }
            
            showAlert(`File loaded successfully. ${data.headers.length} columns found.`, 'success');
        } else {
            showAlert('No headers found in the file.', 'danger');
        }
    })
    .catch(error => {
        console.error('Error processing file:', error);
        
        // Hide loading animation
        document.getElementById('loadingIndicator').style.display = 'none';
        
        showAlert(`Error processing file: ${error.message}`, 'danger');
    });
}

/**
 * Generate checkboxes for each header in the CSV file
 */
function generateHeaderCheckboxes(headers) {
    console.log('Generating header checkboxes for:', headers);
    
    const headerCheckboxesContainer = document.getElementById('headerCheckboxes');
    if (!headerCheckboxesContainer) {
        console.error("Could not find element with ID 'headerCheckboxes'");
        return;
    }
    
    headerCheckboxesContainer.innerHTML = '';
    
    // Create container for checkboxes
    const container = document.createElement('div');
    container.className = 'header-checkboxes-container';
    
    // Add "Select All" checkbox
    const selectAllDiv = document.createElement('div');
    selectAllDiv.className = 'form-check mb-2';
    
    const selectAllCheckbox = document.createElement('input');
    selectAllCheckbox.type = 'checkbox';
    selectAllCheckbox.id = 'selectAllHeaders';
    selectAllCheckbox.className = 'form-check-input';
    
    const selectAllLabel = document.createElement('label');
    selectAllLabel.htmlFor = 'selectAllHeaders';
    selectAllLabel.className = 'form-check-label fw-bold';
    selectAllLabel.textContent = 'Select All';
    
    selectAllDiv.appendChild(selectAllCheckbox);
    selectAllDiv.appendChild(selectAllLabel);
    container.appendChild(selectAllDiv);
    
    // Add horizontal rule
    const hr = document.createElement('hr');
    container.appendChild(hr);
    
    // Add each header as a checkbox
    const headerContainer = document.createElement('div');
    headerContainer.className = 'd-flex flex-column';
    
    headers.forEach(header => {
        const colDiv = document.createElement('div');
        colDiv.className = 'mb-2';
        
        const checkDiv = document.createElement('div');
        checkDiv.className = 'form-check';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `header_${header}`;
        checkbox.className = 'form-check-input header-checkbox';
        checkbox.value = header;
        
        const label = document.createElement('label');
        label.htmlFor = `header_${header}`;
        label.className = 'form-check-label';
        label.textContent = header;
        
        checkDiv.appendChild(checkbox);
        checkDiv.appendChild(label);
        
        // Create a container for additional type options (DateTime and Categorical)
        const extraOptionsDiv = document.createElement('div');
        extraOptionsDiv.className = 'extra-options mt-1 ms-3';
        
        // DateTime checkbox option
        const dateDiv = document.createElement('div');
        dateDiv.className = 'form-check form-check-inline';
        const dateCheckbox = document.createElement('input');
        dateCheckbox.type = 'checkbox';
        dateCheckbox.id = `datetime_${header}`;
        dateCheckbox.className = 'form-check-input datetime-checkbox';
        dateCheckbox.value = header;
        const dateLabel = document.createElement('label');
        dateLabel.htmlFor = `datetime_${header}`;
        dateLabel.className = 'form-check-label';
        dateLabel.textContent = 'DateTime';
        dateDiv.appendChild(dateCheckbox);
        dateDiv.appendChild(dateLabel);
        
        // Categorical checkbox option
        const catDiv = document.createElement('div');
        catDiv.className = 'form-check form-check-inline';
        const catCheckbox = document.createElement('input');
        catCheckbox.type = 'checkbox';
        catCheckbox.id = `categorical_${header}`;
        catCheckbox.className = 'form-check-input categorical-checkbox';
        catCheckbox.value = header;
        const catLabel = document.createElement('label');
        catLabel.htmlFor = `categorical_${header}`;
        catLabel.className = 'form-check-label';
        catLabel.textContent = 'Categorical';
        catDiv.appendChild(catCheckbox);
        catDiv.appendChild(catLabel);
        
        extraOptionsDiv.appendChild(dateDiv);
        extraOptionsDiv.appendChild(catDiv);
        
        colDiv.appendChild(checkDiv);
        colDiv.appendChild(extraOptionsDiv);
        headerContainer.appendChild(colDiv);
        
        console.log(`Created checkboxes for header: ${header} with additional options for DateTime and Categorical`);
    });
    
    container.appendChild(headerContainer);
    headerCheckboxesContainer.appendChild(container);
    
    // Add event listener for "Select All" checkbox
    selectAllCheckbox.addEventListener('change', function() {
        const headerCheckboxes = document.querySelectorAll('.header-checkbox');
        headerCheckboxes.forEach(checkbox => {
            checkbox.checked = selectAllCheckbox.checked;
        });
    });
    
    // Show header selection section
    const headerSelectionSection = document.getElementById('headerSelectionSection');
    if (headerSelectionSection) {
        headerSelectionSection.style.display = 'block';
    }
    
    // Update RMF dropdown options
    if (window.updateRmfColumnOptions) {
        window.updateRmfColumnOptions(headers);
    } else {
        console.warn('updateRmfColumnOptions function not available');
    }
}

/**
 * Display alert messages
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
    const container = document.querySelector('.app-container');
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
 * Analyze selected data and send to server for analysis
 * @param {boolean|Event} isRmfAnalysis Whether this is an RMF analysis or an event object
 * @param {string} recencyColumn Column for RMF recency
 * @param {string} frequencyColumn Column for RMF frequency
 * @param {string} monetaryColumn Column for RMF monetary value
 * @param {boolean} useAdvancedRmf Whether to use advanced RMF
 */
export function analyzeData(isRmfAnalysis = false, recencyColumn = '', frequencyColumn = '', monetaryColumn = '', useAdvancedRmf = false) {
    // Check if first parameter is actually an event object (from direct event binding)
    // If so, treat this as a regular analysis, not RMF
    if (isRmfAnalysis && typeof isRmfAnalysis === 'object' && isRmfAnalysis.type === 'click') {
        console.log('Event object detected, treating as regular analysis');
        isRmfAnalysis = false;
    }
    
    console.log(`analyzeData called with parameters:
        isRmfAnalysis: ${isRmfAnalysis}
        recencyColumn: ${recencyColumn}
        frequencyColumn: ${frequencyColumn}
        monetaryColumn: ${monetaryColumn}
        useAdvancedRmf: ${useAdvancedRmf}
    `);
    
    // RMF functionality is disabled
    if (isRmfAnalysis) {
        console.warn("RMF analysis is disabled");
        showAlert("RMF analysis is currently disabled.", "warning");
        return;
    }
    
    console.log('Analyzing data...', { isRmfAnalysis, recencyColumn, frequencyColumn, monetaryColumn, useAdvancedRmf });
    console.log('Current file:', currentFile);
    
    if (!currentFile) {
        showAlert('Please upload a file first.', 'warning');
        return;
    }
    
    // Show loading animation
    document.getElementById('loadingIndicator').style.display = 'block';
    const analyzeBtn = document.getElementById('analyzeButton');
    if (analyzeBtn) analyzeBtn.disabled = true;
    const rmfButton = document.getElementById('startRmfAnalysis');
    if (rmfButton) rmfButton.disabled = true;
    
    console.log('Getting selected headers...');
    // Get selected headers
    const selectedHeaders = [];
    const headerCheckboxes = document.querySelectorAll('.header-checkbox:checked');
    console.log('Found checked header checkboxes:', headerCheckboxes.length);
    
    if (headerCheckboxes.length === 0) {
        showAlert('Please select at least one header to analyze.', 'danger');
        document.getElementById('loadingIndicator').style.display = 'none';
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (rmfButton) rmfButton.disabled = false;
        return;
    }
    
    headerCheckboxes.forEach(checkbox => {
        console.log('Checkbox:', checkbox.value, checkbox);
        if (checkbox.value !== 'on') { // Skip the "Select All" checkbox
            selectedHeaders.push(checkbox.value);
        }
    });
    
    console.log('Selected headers:', selectedHeaders);
    
    // Additional treatment for DateTime and Categorical checkboxes
    const datetimeHeaders = [];
    const categoricalHeaders = [];
    selectedHeaders.forEach(header => {
        const dtCheckbox = document.getElementById(`datetime_${header}`);
        if (dtCheckbox && dtCheckbox.checked) {
            datetimeHeaders.push(header);
        }
        const catCheckbox = document.getElementById(`categorical_${header}`);
        if (catCheckbox && catCheckbox.checked) {
            categoricalHeaders.push(header);
        }
    });

    console.log('DateTime columns selected:', datetimeHeaders);
    console.log('Categorical columns selected:', categoricalHeaders);
    
    // Create FormData and append selected headers
    const formData = new FormData();
    formData.append('file', currentFile);
    console.log('Added file to FormData:', currentFile.name);
    
    // Then, after appending selected headers to FormData, append the additional type info
    selectedHeaders.forEach(header => {
        formData.append('headers_to_process', header);
        console.log('Added header to FormData:', header);
    });

    datetimeHeaders.forEach(header => {
        formData.append('datetime_columns', header);
        console.log('Added DateTime column to FormData:', header);
    });

    categoricalHeaders.forEach(header => {
        formData.append('categorical_columns', header);
        console.log('Added Categorical column to FormData:', header);
    });
    
    // Add RMF parameters if this is an RMF analysis
    if (isRmfAnalysis) {
        formData.append('run_rmf_analysis', 'true');
        formData.append('recency_col', recencyColumn);
        formData.append('frequency_col', frequencyColumn);
        formData.append('monetary_col', monetaryColumn);
        formData.append('use_advanced_rmf', useAdvancedRmf ? 'true' : 'false');
        
        console.log('RMF parameters added to request', {
            recency_col: recencyColumn,
            frequency_col: frequencyColumn,
            monetary_col: monetaryColumn,
            use_advanced_rmf: useAdvancedRmf
        });
    }
    
    console.log('Sending request to /api/analysis/analyze...');
    // Send request to the server
    fetch('/api/analysis/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Received response:', response.status, response.statusText);
        if (!response.ok) {
            return response.json().then(data => {
                console.error('Error in response:', data);
                throw new Error(data.error || 'Error analyzing data');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Analysis data received:', data);
        
        // Hide loading animation
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        } else {
            console.warn('Loading indicator element not found');
        }
        
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (rmfButton) rmfButton.disabled = false;
        
        // Ensure reportSection exists
        let reportSection = document.getElementById('reportSection');
        if (!reportSection) {
            console.warn('Creating reportSection element');
            reportSection = document.createElement('div');
            reportSection.id = 'reportSection';
            
            // Try to append to results container
            const resultsContainer = document.getElementById('results');
            if (resultsContainer) {
                resultsContainer.appendChild(reportSection);
            } else {
                // Fallback to appending to body
                document.body.appendChild(reportSection);
            }
        }
        
        // Make sure the results container is visible
        const resultsContainer = document.getElementById('results');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
        
        // Check which sections should be displayed based on the data received
        if (data.summary) {
            console.log('Rendering summary stats');
            renderSummaryStats(data.summary);
            
            const summarySection = document.getElementById('summarySection');
            if (summarySection) {
                summarySection.style.display = 'block';
            }
        }
        
        if (data.distributions) {
            console.log('Rendering distributions');
            renderDistributions(data.distributions);
            
            const distributionsSection = document.getElementById('distributionsSection');
            if (distributionsSection) {
                distributionsSection.style.display = 'block';
            }
        }
        
        if (data.correlations) {
            console.log('Rendering correlations');
            renderCorrelations(data.correlations, data.interpretations ? data.interpretations.correlations : null);
            
            const correlationsSection = document.getElementById('correlationsSection');
            if (correlationsSection) {
                correlationsSection.style.display = 'block';
            }
        }
        
        if (data.graphs) {
            console.log('Rendering graphs');
            renderGraphs(data.graphs, data.interpretations || {});
            
            const graphsSection = document.getElementById('graphsSection');
            if (graphsSection) {
                graphsSection.style.display = 'block';
            }
        }
        
        if (data.pairwise_graphs) {
            console.log('Rendering pairwise graphs');
            renderPairwiseGraphs(data.pairwise_graphs, data.interpretations ? data.interpretations.pairwise : {});
            
            const pairwiseGraphsSection = document.getElementById('pairwiseGraphsSection');
            if (pairwiseGraphsSection) {
                pairwiseGraphsSection.style.display = 'block';
            }
        }
        
        // Handle RMF results if present - DISABLED
        /* RMF functionality disabled
        if (data.rmf_results) {
            console.log('Rendering RMF results');
            renderRmf(data.rmf_results);
            
            const rmfSection = document.getElementById('rmfSection');
            if (rmfSection) {
                rmfSection.style.display = 'block';
            }
        }
        */
        
        // Show report section
        reportSection.style.display = 'block';
        
        // Scroll to report section
        try {
            reportSection.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error scrolling to report section:', error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert(error.message || 'Error analyzing data', 'danger');
        
        // Hide loading animation
        document.getElementById('loadingIndicator').style.display = 'none';
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (rmfButton) rmfButton.disabled = false;
    });
} 