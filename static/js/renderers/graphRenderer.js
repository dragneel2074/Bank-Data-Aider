import { toTitleCase } from '../utils/stringUtils.js';

/**
 * Renders graphs for individual columns
 */
export function renderGraphs(graphs) {
    let html = '<div class="graph-grid">';
    for (const [colName, colGraphs] of Object.entries(graphs)) {
        for (const [graphType, graphData] of Object.entries(colGraphs)) {
            if (graphData && typeof graphData === 'string' && !graphData.startsWith("Error") && graphData !== "No data available to plot") {
                html += `
                    <div class="graph-container">
                        <h4>${colName} - ${toTitleCase(graphType.replace('_', ' '))}</h4>
                        <img src="data:image/png;base64,${graphData}"
                             style="width: 100%; height: auto;">
                    </div>
                `;
            } else if (graphData && (graphData.startsWith("Error") || graphData === "No data available to plot")) {
                html += `
                    <div class="graph-container">
                        <h4>${colName} - ${toTitleCase(graphType.replace('_', ' '))}</h4>
                        <p class="error">${graphData}</p>
                    </div>
                `;
            }
        }
    }
    html += '</div>';
    document.getElementById('graphsContent').innerHTML = html;
}

/**
 * Renders pairwise graphs
 */
export function renderPairwiseGraphs(pairwiseGraphs, interpretations = {}) {
    console.log("renderPairwiseGraphs called with:", pairwiseGraphs);
    
    if (!pairwiseGraphs || Object.keys(pairwiseGraphs).length === 0) {
        console.log("No pairwise graphs data available");
        document.getElementById('pairwiseGraphsContent').innerHTML = 
            '<div class="alert alert-info">No pairwise graphs available for the selected columns.</div>';
        return;
    }
    
    try {
        let html = '<div class="graph-grid">';
        for (const col1 in pairwiseGraphs) {
            for (const col2 in pairwiseGraphs[col1]) {
                const graphs = pairwiseGraphs[col1][col2];
                const graphInterpretations = interpretations && interpretations[col1] ? 
                    interpretations[col1][col2] || {} : {};
                
                console.log(`Processing graphs for ${col1} vs ${col2}:`, graphs);
                
                for (const graphType in graphs) {
                    const graphData = graphs[graphType];
                    const interpretation = graphInterpretations[graphType] || '';
                    
                    if (graphData && typeof graphData === 'string' && 
                        !graphData.startsWith("Error") && 
                        !graphData.startsWith("No ")) {
                        html += `
                            <div class="graph-container">
                                <h4>${col1} vs ${col2} - ${toTitleCase(graphType.replace('_', ' '))}</h4>
                                <img src="data:image/png;base64,${graphData}"
                                     style="width: 100%; height: auto;">
                                <p class="interpretation">${interpretation}</p>
                            </div>
                        `;
                    } else if (graphData) {
                        html += `
                            <div class="graph-container">
                                <h4>${col1} vs ${col2} - ${toTitleCase(graphType.replace('_', ' '))}</h4>
                                <p class="error">${graphData}</p>
                            </div>
                        `;
                    }
                }
            }
        }
        html += '</div>';
        document.getElementById('pairwiseGraphsContent').innerHTML = html;
    } catch (error) {
        console.error("Error rendering pairwise graphs:", error);
        document.getElementById('pairwiseGraphsContent').innerHTML = 
            `<div class="alert alert-danger">Error rendering pairwise graphs: ${error.message}</div>`;
    }
} 