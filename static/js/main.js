import { handleFileUpload, analyzeData } from './fileHandler.js';
import { renderGraphs, renderPairwiseGraphs } from './renderers/graphRenderer.js';
import { renderSummaryStats, renderDistributions, renderCorrelations, renderRmf } from './renderers/statsRenderer.js';

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
            console.log('Analyze button clicked');
            // Call analyzeData with standard parameters (not RMF)
            analyzeData(false, '', '', '', false);
        });
    } else {
        console.error('Analyze button element not found');
    }
    
    /* RMF functionality disabled
    // Add event listener for RMF analysis button
    const rmfAnalysisButton = document.getElementById('startRmfAnalysis');
    if (rmfAnalysisButton) {
        rmfAnalysisButton.addEventListener('click', function() {
            // Get RMF columns
            const recencyColumn = document.getElementById('recencyColumn').value;
            const frequencyColumn = document.getElementById('frequencyColumn').value;
            const monetaryColumn = document.getElementById('monetaryColumn').value;
            const useAdvancedRmf = document.getElementById('useAdvancedRmf').checked;
            
            console.log('RMF Analysis requested:', {
                recencyColumn,
                frequencyColumn,
                monetaryColumn,
                useAdvancedRmf
            });
            
            // Call analyzeData with RMF parameters
            analyzeData(true, recencyColumn, frequencyColumn, monetaryColumn, useAdvancedRmf);
        });
    } else {
        console.warn('RMF analysis button not found in the DOM');
    }
    
    // Setup RMF columns dropdowns change event to enable/disable RMF analysis
    setupRmfColumnsListeners();
    */
});

/**
 * Set up event listeners for RMF column selection dropdowns
 */
/* RMF functionality disabled
function setupRmfColumnsListeners() {
    // Check for both old and new IDs
    const rmfColumnPairs = [
        ['recencyCol', 'recencyColumn'],
        ['frequencyCol', 'frequencyColumn'],
        ['monetaryCol', 'monetaryColumn']
    ];
    
    rmfColumnPairs.forEach(([newId, oldId]) => {
        // Try with new ID first, then fall back to old ID
        let dropdown = document.getElementById(newId);
        if (!dropdown) {
            dropdown = document.getElementById(oldId);
        }
        
        if (dropdown) {
            dropdown.addEventListener('change', checkRmfSelections);
        } else {
            console.warn(`RMF dropdown ${newId}/${oldId} not found`);
        }
    });
    
    // Also add listener for the advanced RMF checkbox
    const advancedRmfCheckbox = document.getElementById('useAdvancedRmf');
    if (advancedRmfCheckbox) {
        advancedRmfCheckbox.addEventListener('change', function() {
            console.log('Advanced RMF option changed:', this.checked);
        });
    } else {
        console.warn('Advanced RMF checkbox not found');
    }
}
*/

/**
 * Check if all RMF columns are selected and update UI accordingly
 */
/* RMF functionality disabled
function checkRmfSelections() {
    // Check for both old and new IDs
    const recencyCol = document.getElementById('recencyCol')?.value || document.getElementById('recencyColumn')?.value;
    const frequencyCol = document.getElementById('frequencyCol')?.value || document.getElementById('frequencyColumn')?.value;
    const monetaryCol = document.getElementById('monetaryCol')?.value || document.getElementById('monetaryColumn')?.value;
    const startRmfButton = document.getElementById('startRmfAnalysis');
    
    console.log('RMF column selection changed:', recencyCol, frequencyCol, monetaryCol);
    
    if (startRmfButton) {
        if (recencyCol && frequencyCol && monetaryCol) {
            startRmfButton.disabled = false;
            startRmfButton.title = 'Run RMF analysis with selected columns';
        } else {
            startRmfButton.disabled = true;
            startRmfButton.title = 'Please select all three columns to enable RMF analysis';
        }
    }
}
*/

/**
 * Updates RMF dropdown options after headers are loaded
 */
/* RMF functionality disabled
window.updateRmfColumnOptions = function(headers) {
    // Check for both old and new IDs
    const rmfColumnPairs = [
        ['recencyCol', 'recencyColumn'],
        ['frequencyCol', 'frequencyColumn'],
        ['monetaryCol', 'monetaryColumn']
    ];
    
    rmfColumnPairs.forEach(([newId, oldId]) => {
        // Try with new ID first, then fall back to old ID
        let dropdown = document.getElementById(newId);
        if (!dropdown) {
            dropdown = document.getElementById(oldId);
        }
        
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
        } else {
            console.warn(`RMF dropdown ${newId}/${oldId} not found when updating options`);
        }
    });
    
    // Show RMF section after headers are loaded
    const rmfSetupSection = document.getElementById('rmfSetupSection');
    if (rmfSetupSection) {
        rmfSetupSection.style.display = 'block';
    } else {
        console.warn('RMF setup section not found');
    }
    
    // Initialize button state
    checkRmfSelections();
};
*/

// Global render function used by fileHandler
window.renderReport = function(report) {
    console.log('Rendering report:', report);
    
    if (report.summary) {
        renderSummaryStats(report.summary);
    }
    
    if (report.distributions) {
        renderDistributions(report.distributions);
    }
    
    if (report.graphs) {
        renderGraphs(report.graphs, report.interpretations);
    }
    
    if (report.correlations) {
        renderCorrelations(report.correlations, report.interpretations ? report.interpretations.correlations : null);
    }
    
    // Add pairwise graphs rendering
    if (report.pairwise_graphs) {
        console.log('Rendering pairwise graphs:', report.pairwise_graphs);
        renderPairwiseGraphs(report.pairwise_graphs, report.interpretations ? report.interpretations.pairwise : null);
    }
    
    // Add RMF analysis rendering
    if (report.rmf_results) {
        console.log('Rendering RMF analysis:', report.rmf_results);
        renderRmf(report.rmf_results);
    }
    
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Show pairwise graphs section if it exists
    const pairwiseSection = document.getElementById('pairwiseGraphsSection');
    if (pairwiseSection) {
        pairwiseSection.style.display = 'block';
    }
}; 