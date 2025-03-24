import { handleFileUpload, analyzeData } from './handlers/fileHandler.js';
import { renderGraphs, renderPairwiseGraphs } from './renderers/graphRenderer.js';
import { renderOverview, renderBasicStats, renderCorrelations, renderRmf } from './renderers/statsRenderer.js';

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing event listeners');
    
    // Add file upload event listener
    const fileInput = document.getElementById('csvFile');
    if (fileInput) {
        console.log('Found file input element');
        fileInput.addEventListener('change', handleFileUpload);
    } else {
        console.error('File input element not found');
    }
    
    // Add analyze button event listener
    const analyzeBtn = document.getElementById('analyzeButton');
    if (analyzeBtn) {
        console.log('Found analyze button element');
        analyzeBtn.addEventListener('click', function() {
            // Get RMF column selections if they exist
            const recencyCol = document.getElementById('recencyColumn')?.value;
            const frequencyCol = document.getElementById('frequencyColumn')?.value;
            const monetaryCol = document.getElementById('monetaryColumn')?.value;
            
            console.log(`RMF columns selected - Recency: ${recencyCol}, Frequency: ${frequencyCol}, Monetary: ${monetaryCol}`);
            
            // Pass RMF columns to analyzeData
            analyzeData(recencyCol, frequencyCol, monetaryCol);
        });
    } else {
        console.error('Analyze button element not found');
    }
    
    // Setup RMF columns dropdowns change event to enable/disable RMF analysis
    setupRmfColumnsListeners();
});

/**
 * Set up event listeners for RMF column selection dropdowns
 */
function setupRmfColumnsListeners() {
    const rmfColumns = ['recencyColumn', 'frequencyColumn', 'monetaryColumn'];
    
    rmfColumns.forEach(colId => {
        const dropdown = document.getElementById(colId);
        if (dropdown) {
            dropdown.addEventListener('change', checkRmfSelections);
        }
    });
}

/**
 * Check if all RMF columns are selected and update UI accordingly
 */
function checkRmfSelections() {
    const recencyCol = document.getElementById('recencyColumn')?.value;
    const frequencyCol = document.getElementById('frequencyColumn')?.value;
    const monetaryCol = document.getElementById('monetaryColumn')?.value;
    const rmfInfoElement = document.getElementById('rmfSelectionInfo');
    
    if (rmfInfoElement) {
        if (recencyCol && frequencyCol && monetaryCol) {
            rmfInfoElement.innerHTML = '<div class="alert alert-success">RMF analysis will be performed with selected columns.</div>';
        } else {
            rmfInfoElement.innerHTML = '<div class="alert alert-info">Select all three columns to enable RMF analysis.</div>';
        }
    }
}

/**
 * Updates RMF dropdown options after headers are loaded
 */
window.updateRmfColumnOptions = function(headers) {
    const rmfColumns = ['recencyColumn', 'frequencyColumn', 'monetaryColumn'];
    
    rmfColumns.forEach(colId => {
        const dropdown = document.getElementById(colId);
        if (dropdown) {
            // Clear existing options
            dropdown.innerHTML = '<option value="">Select Column</option>';
            
            // Add options for each header
            headers.forEach(header => {
                const option = document.createElement('option');
                option.value = header;
                option.textContent = header;
                dropdown.appendChild(option);
            });
        }
    });
    
    // Show RMF section after headers are loaded
    const rmfSetupSection = document.getElementById('rmfSetupSection');
    if (rmfSetupSection) {
        rmfSetupSection.style.display = 'block';
    }
    
    // Initialize info message
    checkRmfSelections();
}

// Global render function used by fileHandler
window.renderReport = function(report) {
    console.log('Rendering report:', report);
    
    if (report.overview) {
        renderOverview(report.overview);
    }
    
    if (report.basic_stats) {
        renderBasicStats(report.basic_stats);
    }
    
    if (report.graphs) {
        renderGraphs(report.graphs, report.graphs_interpretations);
    }
    
    if (report.correlations) {
        renderCorrelations(
            report.correlations,
            report.correlation_interpretations?.standout_interpretation
        );
    }
    
    // Add pairwise graphs rendering
    if (report.pairwise_graphs) {
        console.log('Rendering pairwise graphs:', report.pairwise_graphs);
        renderPairwiseGraphs(report.pairwise_graphs, report.pairwise_graphs_interpretations);
    }
    
    // Add RMF analysis rendering
    if (report.rmf_analysis) {
        console.log('Rendering RMF analysis:', report.rmf_analysis);
        renderRmf(report.rmf_analysis);
        
        // Show RMF section
        const rmfSection = document.getElementById('rmfSection');
        if (rmfSection) {
            rmfSection.style.display = 'block';
        }
    }
    
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Show pairwise graphs section if it exists
    const pairwiseSection = document.getElementById('pairwiseGraphsSection');
    if (pairwiseSection) {
        pairwiseSection.style.display = 'block';
    }
}; 