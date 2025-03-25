import { toTitleCase } from '../utils/stringUtils.js';

/**
 * Renders summary statistics
 */
export function renderSummaryStats(summary) {
    console.log("Rendering summary stats", summary);
    
    let summaryContentElement = document.getElementById('summaryContent');
    if (!summaryContentElement) {
        console.error("Could not find element with ID 'summaryContent', creating one.");
        let reportSection = document.getElementById('reportSection');
        if (!reportSection) {
            console.warn("Could not find 'reportSection'; appending to body.");
            reportSection = document.createElement('div');
            reportSection.id = 'reportSection';
            document.body.appendChild(reportSection);
        }
        summaryContentElement = document.createElement('div');
        summaryContentElement.id = 'summaryContent';
        reportSection.appendChild(summaryContentElement);
    }
    
    let html = `
        <div class="card mb-4">
            <div class="card-header">
                <h3>File Summary</h3>
            </div>
            <div class="card-body">
                <p><strong>File Name:</strong> ${summary.file_name}</p>
                <p><strong>Total Rows:</strong> ${summary.total_rows}</p>
                <p><strong>Total Columns:</strong> ${summary.total_columns}</p>
                <p><strong>Columns Analyzed:</strong> ${summary.columns_analyzed.join(', ')}</p>
            </div>
        </div>
    `;
    
    summaryContentElement.innerHTML = html;
    document.getElementById('summarySection').style.display = 'block';
}

/**
 * Renders distribution statistics
 */
export function renderDistributions(distributions) {
    console.log("Rendering distributions", distributions);
    
    const distributionsContentElement = document.getElementById('distributionsContent');
    
    if (!distributionsContentElement) {
        console.error("Could not find element with ID 'distributionsContent'");
        return;
    }
    
    let html = '<div class="row">';
    
    for (const [colName, colData] of Object.entries(distributions)) {
        if (colData.error) {
            html += `
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h3>${colName}</h3>
                        </div>
                        <div class="card-body">
                            <div class="alert alert-danger">
                                ${colData.error}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            continue;
        }
        
        html += `
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h3>${colName} ${colData.dtype ? `(${colData.dtype})` : ''}</h3>
                    </div>
                    <div class="card-body">
                        <table class="table table-sm table-striped">
                            <tbody>
        `;
        
        // Add basic stats with user-friendly labels
        const statLabels = {
            'count': 'Count',
            'mean': 'Mean',
            'std': 'Standard Deviation',
            'min': 'Minimum',
            'max': 'Maximum',
            'median': 'Median',
            'mode': 'Mode',
            'missing': 'Missing Values',
            'unique': 'Unique Values',
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis',
            'q1': '25th Percentile',
            'q3': '75th Percentile',
            'iqr': 'Interquartile Range'
        };
        
        for (const [key, value] of Object.entries(colData)) {
            // Skip certain keys that aren't basic stats
            if (['dtype', 'recommended_tests', 'transform_recommendations', 'is_normal', 'p_value', 'error'].includes(key)) {
                continue;
            }
            
            const label = statLabels[key] || toTitleCase(key.replace(/_/g, ' '));
            html += `
                <tr>
                    <td>${label}</td>
                    <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                </tr>
            `;
        }
        
        html += `
                            </tbody>
                        </table>
        `;
        
        // Add distribution information if available
        if (colData.is_normal !== undefined) {
            html += `
                <div class="mt-3">
                    <p>Distribution: 
                        <span class="badge ${colData.is_normal ? 'bg-success' : 'bg-warning'}">
                            ${colData.is_normal ? 'Normal' : 'Non-Normal'}
                        </span>
                        ${colData.p_value !== undefined ? `(p-value: ${colData.p_value.toFixed(4)})` : ''}
                    </p>
                </div>
            `;
        }
        
        // Add test recommendations if available
        if (colData.recommended_tests) {
            html += `
                <div class="mt-3">
                    <p><strong>Recommended Tests:</strong></p>
                    <ul>
                        ${colData.recommended_tests.correlation ? `<li>Correlation: ${colData.recommended_tests.correlation}</li>` : ''}
                        ${colData.recommended_tests.group_comparison ? `<li>Group Comparison: ${colData.recommended_tests.group_comparison}</li>` : ''}
                    </ul>
                </div>
            `;
        }
        
        // Add transformation recommendations if available
        if (colData.transform_recommendations && colData.transform_recommendations.length > 0) {
            html += `
                <div class="mt-3">
                    <p><strong>Suggested Transformations:</strong></p>
                    <ul>
                        ${colData.transform_recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    distributionsContentElement.innerHTML = html;
    document.getElementById('distributionsSection').style.display = 'block';
}

/**
 * Renders correlation analysis
 */
export function renderCorrelations(correlations, interpretations) {
    console.log("Rendering correlations", correlations);
    
    const correlationContentElement = document.getElementById('correlationContent');
    
    if (!correlationContentElement) {
        console.error("Could not find element with ID 'correlationContent'");
        return;
    }
    
    if (correlations.error) {
        correlationContentElement.innerHTML = `
            <div class="alert alert-danger">
                ${correlations.error}
            </div>
        `;
        return;
    }
    
    let html = '<div class="table-responsive"><table class="table table-striped table-bordered">';
    html += `
        <thead>
            <tr>
                <th>Feature Pair</th>
                <th>Analysis Type</th>
                <th>Correlation</th>
                <th>p-value</th>
                <th>Significance</th>
            </tr>
        </thead>
        <tbody>
    `;

    for (const [pair, data] of Object.entries(correlations)) {
        // Determine correlation strength class
        let correlationClass = '';
        let correlationText = '';
        
        if (data.coefficient !== undefined) {
            const absCoef = Math.abs(data.coefficient);
            if (absCoef >= 0.7) {
                correlationClass = 'table-danger';
                correlationText = 'Strong';
            } else if (absCoef >= 0.4) {
                correlationClass = 'table-warning';
                correlationText = 'Moderate';
            } else if (absCoef >= 0.2) {
                correlationClass = 'table-info';
                correlationText = 'Weak';
            } else {
                correlationClass = 'table-light';
                correlationText = 'Very Weak';
            }
        }
        
        // Determine significance
        let sigText = 'Not Significant';
        if (data.p_value !== undefined && data.p_value < 0.05) {
            sigText = data.p_value < 0.001 ? 'Highly Significant (p<0.001)' : 
                      data.p_value < 0.01 ? 'Very Significant (p<0.01)' : 'Significant (p<0.05)';
        }
        
        html += `
            <tr class="${correlationClass}">
                <td>${pair}</td>
                <td>${data.type || 'Unknown'}</td>
                <td>${data.coefficient !== undefined ? data.coefficient.toFixed(4) : 
                      data.statistic !== undefined ? data.statistic.toFixed(4) : 'N/A'}
                </td>
                <td>${data.p_value !== undefined ? data.p_value.toFixed(4) : 'N/A'}</td>
                <td>${sigText}${correlationText ? ` (${correlationText})` : ''}</td>
            </tr>
        `;
    }

    html += '</tbody></table></div>';
    
    // Add interpretation text if available
    if (interpretations && interpretations.correlations) {
        html += `
            <div class="card mt-4">
                <div class="card-header">
                    <h3>Interpretation</h3>
                </div>
                <div class="card-body">
                    <p>${interpretations.correlations}</p>
                </div>
            </div>
        `;
    }
    
    correlationContentElement.innerHTML = html;
    document.getElementById('correlationsSection').style.display = 'block';
}

/**
 * Renders RMF (Recency, Frequency, Monetary) analysis results
 */
export function renderRmf(rmfData) {
    console.log("Rendering RMF results", rmfData);
    
    const rmfContentElement = document.getElementById('rmfContent');
    
    if (!rmfContentElement) {
        console.error("Could not find element with ID 'rmfContent'");
        return;
    }
    
    if (rmfData.error) {
        rmfContentElement.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error calculating RMF:</strong> ${rmfData.error}
            </div>
        `;
        return;
    }
    
    if (rmfData.message) {
        rmfContentElement.innerHTML = `
            <div class="alert alert-info">
                ${rmfData.message}
            </div>
        `;
        return;
    }
    
    // Create the method description
    let methodDescription = "";
    if (rmfData.method) {
        methodDescription = `<div class="alert alert-info">
            <strong>Analysis Method:</strong> ${rmfData.method}
            ${rmfData.has_clustering ? 
                '<div class="mt-2"><span class="badge bg-primary">Advanced</span> Using K-means clustering for monetary values to identify natural spending patterns.</div>' 
                : ''}
        </div>`;
    }
    
    // Create the segment summary
    let segmentSummary = '<div class="segment-summary mb-4"><h4>Customer Segments</h4><ul class="list-group">';
    for (const [segment, count] of Object.entries(rmfData.segment_summary)) {
        segmentSummary += `
            <li class="list-group-item d-flex justify-content-between align-items-center">
                ${segment}
                <span class="badge bg-primary rounded-pill">${count}</span>
            </li>
        `;
    }
    segmentSummary += '</ul></div>';
    
    // Create the interpretation section
    let interpretationSection = '';
    if (rmfData.interpretation) {
        interpretationSection = `
            <div class="card mb-4">
                <div class="card-header">
                    <h3>Interpretation Guide</h3>
                </div>
                <div class="card-body">
                    <p><strong>R Score:</strong> ${rmfData.interpretation.r_score}</p>
                    <p><strong>F Score:</strong> ${rmfData.interpretation.f_score}</p>
                    <p><strong>M Score:</strong> ${rmfData.interpretation.m_score}</p>
                    <p><strong>Segments:</strong> ${rmfData.interpretation.segments}</p>
                    ${rmfData.interpretation.clustering ? `<p><strong>Clustering:</strong> ${rmfData.interpretation.clustering}</p>` : ''}
                </div>
            </div>
        `;
    }
    
    // Create the columns used section
    const columnsUsed = `
        <div class="mb-4">
            <h4>Columns Used</h4>
            <ul class="list-group">
                <li class="list-group-item"><strong>Recency:</strong> ${rmfData.columns_used.recency}</li>
                <li class="list-group-item"><strong>Frequency:</strong> ${rmfData.columns_used.frequency}</li>
                <li class="list-group-item"><strong>Monetary:</strong> ${rmfData.columns_used.monetary}</li>
            </ul>
        </div>
    `;
    
    // Create the data table
    let dataTable = '<h4 class="mt-4">RMF Analysis Results</h4>';
    dataTable += '<div class="table-responsive"><table class="table table-striped table-bordered">';
    
    // Table headers
    dataTable += '<thead><tr>';
    if (rmfData.data && rmfData.data.length > 0) {
        const keys = Object.keys(rmfData.data[0]);
        keys.forEach(key => {
            dataTable += `<th>${key}</th>`;
        });
    }
    dataTable += '</tr></thead>';
    
    // Table body
    dataTable += '<tbody>';
    if (rmfData.data && rmfData.data.length > 0) {
        // Only show first 100 rows
        const displayData = rmfData.data.slice(0, 100);
        displayData.forEach(row => {
            dataTable += '<tr>';
            Object.values(row).forEach(value => {
                dataTable += `<td>${value}</td>`;
            });
            dataTable += '</tr>';
        });
        
        if (rmfData.data.length > 100) {
            dataTable += `<tr><td colspan="${Object.keys(rmfData.data[0]).length}" class="text-center">
                <em>Showing 100 of ${rmfData.data.length} rows</em>
            </td></tr>`;
        }
    }
    dataTable += '</tbody></table></div>';
    
    // Put everything together
    rmfContentElement.innerHTML = `
        ${methodDescription}
        <div class="row">
            <div class="col-md-6">
                ${columnsUsed}
            </div>
            <div class="col-md-6">
                ${segmentSummary}
            </div>
        </div>
        ${interpretationSection}
        ${dataTable}
    `;
    
    // Make sure the section is visible
    const rmfSection = document.getElementById('rmfSection');
    if (rmfSection) {
        rmfSection.style.display = 'block';
    } else {
        console.error("Could not find element with ID 'rmfSection'");
    }
} 