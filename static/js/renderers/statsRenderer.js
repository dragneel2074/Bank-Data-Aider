import { toTitleCase } from '../utils/stringUtils.js';

/**
 * Renders overview statistics
 */
export function renderOverview(overview) {
    let html = '';
    for (const [colName, colData] of Object.entries(overview)) {
        html += `
            <div class="column-card">
                <h3>${colName} (${colData.dtype})</h3>
                ${colData.is_normal !== undefined ? `
                    <p>Distribution:
                        <span class="badge ${colData.is_normal ? 'normal' : 'not-normal'}">
                            ${colData.is_normal ? 'Normal' : 'Non-Normal'}
                        </span>
                        (p-value: ${colData.p_value?.toFixed(4) ?? 'N/A'})
                    </p>
                    <p>Recommended Tests: ${colData.recommended_tests?.correlation},
                    ${colData.recommended_tests?.group_comparison}</p>
                ` : ''}
                ${colData.transform_recommendations ? `
                    <p>Suggested Transformations:
                    ${colData.transform_recommendations.join(', ')}</p>
                ` : ''}
            </div>
        `;
    }
    document.getElementById('overviewContent').innerHTML = html;
}

/**
 * Renders basic statistics
 */
export function renderBasicStats(stats) {
    let html = '';
    for (const [colName, colData] of Object.entries(stats)) {
        html += `
            <div class="column-card">
                <h3>${colName}</h3>
                <table>
                    <tbody>
                        ${Object.entries(colData).map(([key, value]) => `
                            <tr>
                                <td>${key}</td>
                                <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }
    document.getElementById('basicStatsContent').innerHTML = html;
}

/**
 * Renders correlation analysis
 */
export function renderCorrelations(correlations, interpretationText) {
    let html = '<table>';
    html += `
        <tr>
            <th>Feature Pair</th>
            <th>Analysis Type</th>
            <th>Correlation</th>
            <th>p-value</th>
        </tr>
    `;

    for (const [pair, data] of Object.entries(correlations)) {
        const isStandout = data.is_standout ? 'standout-correlation' : '';
        html += `
            <tr>
                <td>${pair}</td>
                <td>${data.type}</td>
                <td class="${isStandout}">${data.coefficient?.toFixed(4) ?? data.statistic?.toFixed(4) ?? 'N/A'}</td>
                <td>${data.p_value?.toFixed(4) ?? 'N/A'}</td>
            </tr>
        `;
    }

    html += '</table>';
    document.getElementById('correlationContent').innerHTML = html;
    document.getElementById('correlationInterpretation').innerText = interpretationText || '';
}

/**
 * Renders RMF (Recency, Frequency, Monetary) analysis results
 */
export function renderRmf(rmfAnalysis) {
    console.log('Rendering RMF analysis:', rmfAnalysis);
    
    const rmfSection = document.getElementById('rmfContent');
    if (!rmfSection) {
        console.error('RMF section not found in the document');
        return;
    }
    
    // Handle error or message case
    if (rmfAnalysis.error) {
        rmfSection.innerHTML = `<div class="alert alert-danger">${rmfAnalysis.error}</div>`;
        return;
    }
    
    if (rmfAnalysis.message) {
        rmfSection.innerHTML = `<div class="alert alert-info">${rmfAnalysis.message}</div>`;
        return;
    }
    
    try {
        // Create the content for RMF analysis
        let html = '<div class="rmf-container">';
        
        // Add column information
        html += `<div class="rmf-columns mb-4">
            <h4>RMF Analysis Columns</h4>
            <ul class="list-group">
                <li class="list-group-item">Recency: ${rmfAnalysis.columns_used?.recency || 'N/A'}</li>
                <li class="list-group-item">Frequency: ${rmfAnalysis.columns_used?.frequency || 'N/A'}</li>
                <li class="list-group-item">Monetary: ${rmfAnalysis.columns_used?.monetary || 'N/A'}</li>
            </ul>
        </div>`;
        
        // Add interpretation
        html += `<div class="rmf-interpretation mb-4">
            <h4>Interpretation</h4>
            <div class="card">
                <div class="card-body">
                    <p><strong>R Score:</strong> ${rmfAnalysis.interpretation?.r_score || 'N/A'}</p>
                    <p><strong>F Score:</strong> ${rmfAnalysis.interpretation?.f_score || 'N/A'}</p>
                    <p><strong>M Score:</strong> ${rmfAnalysis.interpretation?.m_score || 'N/A'}</p>
                    <p><strong>Segments:</strong> ${rmfAnalysis.interpretation?.segments || 'N/A'}</p>
                </div>
            </div>
        </div>`;
        
        // Add segment summary if available
        if (rmfAnalysis.segment_summary && Object.keys(rmfAnalysis.segment_summary).length > 0) {
            html += `<div class="rmf-segments mb-4">
                <h4>Customer Segments</h4>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Segment</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>`;
            
            // Calculate total
            const total = Object.values(rmfAnalysis.segment_summary).reduce((a, b) => a + b, 0);
            
            // Create rows for each segment
            for (const [segment, count] of Object.entries(rmfAnalysis.segment_summary)) {
                const percentage = ((count / total) * 100).toFixed(2);
                html += `<tr>
                    <td>${segment}</td>
                    <td>${count}</td>
                    <td>${percentage}%</td>
                </tr>`;
            }
            
            html += `</tbody></table></div>`;
        }
        
        // Add data table if available
        if (rmfAnalysis.data && rmfAnalysis.data.length > 0) {
            // Get column names from first data object
            const columns = Object.keys(rmfAnalysis.data[0]);
            
            html += `<div class="rmf-data mb-4">
                <h4>RMF Data (Sample)</h4>
                <div class="table-responsive">
                    <table class="table table-sm table-striped">
                        <thead>
                            <tr>`;
                            
            // Add table headers
            columns.forEach(col => {
                html += `<th>${toTitleCase(col)}</th>`;
            });
            
            html += `</tr></thead><tbody>`;
            
            // Add table rows (limit to first 10 for performance)
            const sampleData = rmfAnalysis.data.slice(0, 10);
            sampleData.forEach(row => {
                html += `<tr>`;
                columns.forEach(col => {
                    html += `<td>${row[col] !== undefined ? row[col] : 'N/A'}</td>`;
                });
                html += `</tr>`;
            });
            
            // Show message if there's more data
            if (rmfAnalysis.data.length > 10) {
                const remaining = rmfAnalysis.data.length - 10;
                html += `<tr><td colspan="${columns.length}" class="text-center">... and ${remaining} more rows</td></tr>`;
            }
            
            html += `</tbody></table></div>`;
        }
        
        html += '</div>';
        rmfSection.innerHTML = html;
        
    } catch (error) {
        console.error('Error rendering RMF analysis:', error);
        rmfSection.innerHTML = `<div class="alert alert-danger">Error rendering RMF analysis: ${error.message}</div>`;
    }
} 