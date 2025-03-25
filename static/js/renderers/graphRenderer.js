import { toTitleCase } from '../utils/stringUtils.js';

/**
 * Renders graphs for individual columns
 */
export function renderGraphs(graphs) {
    console.log("Rendering graphs", graphs);
    
    // Check if graphsContent exists, create it if not
    let graphsContent = document.getElementById('graphsContent');
    if (!graphsContent) {
        console.error("Could not find element with ID 'graphsContent', creating one");
        let reportSection = document.getElementById('reportSection');
        if (!reportSection) {
            console.warn("Could not find 'reportSection'; appending to body.");
            reportSection = document.createElement('div');
            reportSection.id = 'reportSection';
            document.body.appendChild(reportSection);
        }
        
        // Create a graphsSection if it doesn't exist
        let graphsSection = document.getElementById('graphsSection');
        if (!graphsSection) {
            console.warn("Creating graphsSection container");
            graphsSection = document.createElement('div');
            graphsSection.id = 'graphsSection';
            graphsSection.className = 'card mt-4';
            graphsSection.innerHTML = `
                <div class="card-header bg-primary text-white">
                    <h3 class="card-title mb-0">Visualizations</h3>
                </div>
                <div class="card-body">
                    <!-- Graphs content will be placed here -->
                </div>
            `;
            reportSection.appendChild(graphsSection);
        }
        
        // Now create the graphsContent element
        graphsContent = document.createElement('div');
        graphsContent.id = 'graphsContent';
        
        // Find the card-body inside graphsSection
        const cardBody = graphsSection.querySelector('.card-body');
        if (cardBody) {
            cardBody.appendChild(graphsContent);
        } else {
            graphsSection.appendChild(graphsContent);
        }
    }

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
    graphsContent.innerHTML = html;
    
    // Show the graphs section
    const graphsSection = document.getElementById('graphsSection');
    if (graphsSection) {
        graphsSection.style.display = 'block';
    }
}

/**
 * Renders pairwise graphs
 */
export function renderPairwiseGraphs(pairwiseGraphs, interpretations = {}) {
    console.log("renderPairwiseGraphs called with:", pairwiseGraphs);
    
    // Check if pairwiseGraphsContent exists, create it if not
    let pairwiseGraphsContent = document.getElementById('pairwiseGraphsContent');
    if (!pairwiseGraphsContent) {
        console.error("Could not find element with ID 'pairwiseGraphsContent', creating one");
        let reportSection = document.getElementById('reportSection');
        if (!reportSection) {
            console.warn("Could not find 'reportSection'; appending to body.");
            reportSection = document.createElement('div');
            reportSection.id = 'reportSection';
            document.body.appendChild(reportSection);
        }
        
        // Create pairwiseGraphsSection if it doesn't exist
        let pairwiseGraphsSection = document.getElementById('pairwiseGraphsSection');
        if (!pairwiseGraphsSection) {
            console.warn("Creating pairwiseGraphsSection container");
            pairwiseGraphsSection = document.createElement('div');
            pairwiseGraphsSection.id = 'pairwiseGraphsSection';
            pairwiseGraphsSection.className = 'card mt-4';
            pairwiseGraphsSection.innerHTML = `
                <div class="card-header bg-primary text-white">
                    <h3 class="card-title mb-0">Pairwise Feature Analysis</h3>
                </div>
                <div class="card-body">
                    <!-- Pairwise graphs content will be placed here -->
                </div>
            `;
            reportSection.appendChild(pairwiseGraphsSection);
        }
        
        // Create the pairwiseGraphsContent element
        pairwiseGraphsContent = document.createElement('div');
        pairwiseGraphsContent.id = 'pairwiseGraphsContent';
        
        // Find the card-body inside pairwiseGraphsSection
        const cardBody = pairwiseGraphsSection.querySelector('.card-body');
        if (cardBody) {
            cardBody.appendChild(pairwiseGraphsContent);
        } else {
            pairwiseGraphsSection.appendChild(pairwiseGraphsContent);
        }
    }
    
    if (!pairwiseGraphs || Object.keys(pairwiseGraphs).length === 0) {
        console.log("No pairwise graphs data available");
        pairwiseGraphsContent.innerHTML = 
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
        pairwiseGraphsContent.innerHTML = html;
        
        // Show the pairwise graphs section
        const pairwiseGraphsSection = document.getElementById('pairwiseGraphsSection');
        if (pairwiseGraphsSection) {
            pairwiseGraphsSection.style.display = 'block';
        }
    } catch (error) {
        console.error("Error rendering pairwise graphs:", error);
        pairwiseGraphsContent.innerHTML = 
            `<div class="alert alert-danger">Error rendering pairwise graphs: ${error.message}</div>`;
    }
} 