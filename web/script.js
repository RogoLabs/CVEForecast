/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions
 */

// Global variables
let forecastData = null;
let chartInstance = null;
let selectedModel = 'combined';
let chartType = 'cumulative';
const TOP_MODELS_COUNT = 5;

// SINGLE SOURCE OF TRUTH: Pre-calculated forecast totals object
let forecastTotals = {};

/**
 * Calculate all forecast totals once and store in forecastTotals object
 * This is the SINGLE SOURCE OF TRUTH for all UI components
 */
function calculateAllForecastTotals() {
    if (!forecastData || !forecastData.historical_data) {
        console.error('No forecast data available for calculations');
        return;
    }
    
    // Clear previous calculations
    forecastTotals = {};
    
    // Step 1: Calculate base historical + current month total
    let baseHistoricalTotal = 0;
    
    // Sum 2025 historical data only
    forecastData.historical_data.forEach(item => {
        if (item.date && item.date.startsWith('2025')) {
            baseHistoricalTotal += item.cve_count;
        }
    });
    
    // Current month actual is for visual reference only - NOT included in yearly forecast totals
    // (Current month data point remains visible on chart but doesn't count toward yearly total)
    
    // Step 2: Identify the best performing model (#1 ranked)
    let bestModelName = 'Unknown';
    if (forecastData.model_rankings && forecastData.model_rankings.length > 0) {
        bestModelName = forecastData.model_rankings[0].model_name;
    }
    
    // Step 3: Calculate individual model totals
    if (forecastData.forecasts) {
        const currentMonth = new Date().getMonth() + 1;
        
        Object.keys(forecastData.forecasts).forEach(modelName => {
            const modelForecasts = forecastData.forecasts[modelName];
            let modelForecastTotal = 0;
            
            if (modelForecasts) {
                modelForecasts.forEach(forecast => {
                    const [year, month] = forecast.date.split('-').map(Number);
                    // Only include future months to avoid double-counting current month
                    if (month > currentMonth || year > 2025) {
                        modelForecastTotal += forecast.cve_count;
                    }
                });
            }
            
            // Store complete total: historical + current + forecast
            forecastTotals[modelName] = Math.round(baseHistoricalTotal + modelForecastTotal);
        });
    }
    
    // Step 4: Calculate "All Models" average
    const modelNames = Object.keys(forecastData.forecasts || {});
    if (modelNames.length > 0) {
        const currentMonth = new Date().getMonth() + 1;
        const monthlyAverages = {};
        
        // Group forecasts by month to calculate averages
        modelNames.forEach(modelName => {
            const modelForecasts = forecastData.forecasts[modelName];
            if (modelForecasts) {
                modelForecasts.forEach(forecast => {
                    const [year, month] = forecast.date.split('-').map(Number);
                    // Only include future months
                    if (month > currentMonth || year > 2025) {
                        if (!monthlyAverages[forecast.date]) {
                            monthlyAverages[forecast.date] = [];
                        }
                        monthlyAverages[forecast.date].push(forecast.cve_count);
                    }
                });
            }
        });
        
        // Calculate average for each month and sum
        let allModelsAverageTotal = 0;
        Object.keys(monthlyAverages).forEach(monthKey => {
            const monthValues = monthlyAverages[monthKey];
            const monthAverage = monthValues.reduce((sum, val) => sum + val, 0) / monthValues.length;
            allModelsAverageTotal += monthAverage;
        });
        
        forecastTotals['all_models_average'] = Math.round(baseHistoricalTotal + allModelsAverageTotal);
    }
    
    // Step 5: Store best model total separately for dashboard card
    if (bestModelName !== 'Unknown' && forecastTotals[bestModelName]) {
        forecastTotals['best_model_total'] = forecastTotals[bestModelName];
        forecastTotals['best_model_name'] = bestModelName;
    }
    
    console.log('📊 Forecast Totals Calculated:', forecastTotals);
    
    // Update debug panel if it exists
    updateDebugPanel();
}

/**
 * Update debug panel with backend pre-calculated forecast totals
 */
function updateDebugPanel() {
    const debugPanel = document.getElementById('debugPanel');
    if (!debugPanel) return;
    
    // Use backend pre-calculated totals instead of JavaScript calculations
    const backendTotals = forecastData.yearly_forecast_totals || {};
    
    // Display the backend pre-calculated totals
    debugPanel.innerHTML = `
        <h3 style="margin: 0 0 10px 0; color: #1f2937; font-weight: bold;">🔧 Debug Panel - Backend Pre-Calculated Totals</h3>
        <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px;">Single source of truth from Python backend (cve_forecast.py):</p>
        <pre style="background: #f9fafb; padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 12px; overflow-x: auto; margin: 0;">${JSON.stringify(backendTotals, null, 2)}</pre>
    `;
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadForecastData();
});

/**
 * Load forecast data from JSON file
 */
async function loadForecastData() {
    console.log('🔄 Starting to load forecast data...');
    try {
        console.log('📡 Fetching data.json...');
        const response = await fetch('data.json');
        console.log('📡 Fetch response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        console.log('📋 Parsing JSON data...');
        forecastData = await response.json();
        console.log('✅ Forecast data loaded successfully. Keys:', Object.keys(forecastData));
        console.log('🔍 Backend data check - yearly_forecast_totals:', forecastData.yearly_forecast_totals);
        console.log('🔍 Backend data check - cumulative_timelines:', Object.keys(forecastData.cumulative_timelines || {}));
        
        console.log('🎨 Updating UI elements...');
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');
        
        console.log('🚀 Starting dashboard initialization...');
        initializeDashboard();
        console.log('✅ Dashboard initialization completed successfully!');
        
    } catch (error) {
        console.error('❌ Error loading forecast data:', error);
        console.error('❌ Error details:', {
            message: error.message,
            stack: error.stack,
            name: error.name
        });
        
        // Show error state
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('errorState').classList.remove('hidden');
    }
}

/**
 * Initialize dashboard components
 */
function initializeDashboard() {
    // Use backend pre-calculated data (no JavaScript calculations needed)
    console.log('Initializing dashboard with backend data:', forecastData.yearly_forecast_totals);
    
    updateSummaryCards();
    populateModelSelector();
    populateValidationModelSelector();
    populateModelRankings();
    populateValidationTable();
    updateDataPeriodInfo();
    updateTimeRangeSelector();
    createChart();
    
    // Add event listeners (handle missing elements from simplified interface)
    const modelSelector = document.getElementById('modelSelector');
    if (modelSelector) {
        modelSelector.addEventListener('change', updateChart);
    }
    
    const timeRange = document.getElementById('timeRange');
    if (timeRange) {
        timeRange.addEventListener('change', updateChart);
    }
    
    const validationModelSelector = document.getElementById('validationModelSelector');
    if (validationModelSelector) {
        validationModelSelector.addEventListener('change', populateValidationTable);
    }
    
    const chartType = document.getElementById('chartType');
    if (chartType) {
        chartType.addEventListener('change', updateChart);
    }
    
    console.log('✅ Simplified interface initialized - using main chart view only');
}

/**
 * Update summary cards with key metrics
 */
function updateSummaryCards() {
    const summary = forecastData.summary;
    
    // Update last updated time
    const lastUpdated = new Date(forecastData.generated_at);
    document.getElementById('lastUpdated').textContent = 
        `Last Updated: ${lastUpdated.toLocaleDateString()} ${lastUpdated.toLocaleTimeString()}`;
    
    // Update total CVEs
    document.getElementById('totalCVEs').textContent = 
        summary.total_historical_cves.toLocaleString();
    
    // Calculate and update current year cumulative forecast
    updateCurrentYearForecast();
    
    // Update best model info
    if (forecastData.model_rankings.length > 0) {
        const bestModel = forecastData.model_rankings[0];
        document.getElementById('bestModel').textContent = bestModel.model_name;
        document.getElementById('bestAccuracy').textContent = bestModel.mape.toFixed(2) + '%';
    }
    
    // Current month progress is now displayed in the chart, not as a separate card
}

/**
 * Populate model selector dropdown (only top 5 models for chart performance)
 * NOTE: Simplified interface - dropdown removed, function disabled
 */
function populateModelSelector() {
    const selector = document.getElementById('modelSelector');
    
    // Handle case where dropdown was removed during simplification
    if (!selector) {
        console.log('✅ Model selector not found - using simplified interface with main chart only');
        return;
    }
    
    // Clear existing options except 'All Models'
    selector.innerHTML = '<option value="all">All Models</option>';
    
    // Only show top 5 models in the chart selector for performance
    if (forecastData.model_rankings && forecastData.model_rankings.length > 0) {
        const top5Models = forecastData.model_rankings.slice(0, 5);
        top5Models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_name;
            option.textContent = model.model_name;
            selector.appendChild(option);
        });
    }
}

/**
 * Populate validation model selector dropdown
 */
function populateValidationModelSelector() {
    const selector = document.getElementById('validationModelSelector');
    
    // Clear existing options
    selector.innerHTML = '';
    
    // Add options for each model ordered by accuracy (best first)
    if (forecastData.model_rankings && forecastData.model_rankings.length > 0) {
        forecastData.model_rankings.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_name;
            option.textContent = model.model_name;
            selector.appendChild(option);
        });
        
        // Set the best model as default (first in rankings)
        selector.value = forecastData.model_rankings[0].model_name;
    }
}

/**
 * Populate model rankings table
 */
function populateModelRankings() {
    const tableBody = document.getElementById('modelRankingsTable');
    tableBody.innerHTML = '';
    
    // Model information links
    const modelLinks = {
        'XGBoost': 'https://en.wikipedia.org/wiki/XGBoost',
        'CatBoost': 'https://catboost.ai/',
        'NHiTS': 'https://arxiv.org/abs/2201.12886',
        'DLinear': 'https://arxiv.org/abs/2205.13504', 
        'KalmanFilter': 'https://en.wikipedia.org/wiki/Kalman_filter',
        'TBATS': 'https://en.wikipedia.org/wiki/TBATS_model',
        'RandomForest': 'https://en.wikipedia.org/wiki/Random_forest',
        'NBEATS': 'https://arxiv.org/abs/1905.10437',
        'TCN': 'https://arxiv.org/abs/1803.01271',
        'NLinear': 'https://arxiv.org/abs/2205.13504',
        'LightGBM': 'https://lightgbm.readthedocs.io/',
        'NaiveDrift': 'https://otexts.com/fpp3/simple-methods.html#drift-method',
        'NaiveMovingAverage': 'https://en.wikipedia.org/wiki/Moving_average',
        'Theta': 'https://en.wikipedia.org/wiki/Theta_model',
        'FourTheta': 'https://github.com/Nixtla/statsforecast/blob/main/statsforecast/models.py#L1427',
        'Prophet': 'https://facebook.github.io/prophet/',
        'Croston': 'https://en.wikipedia.org/wiki/Croston%27s_method',
        'NaiveSeasonal': 'https://otexts.com/fpp3/simple-methods.html#seasonal-naïve-method',
        'NaiveMean': 'https://otexts.com/fpp3/simple-methods.html#average-method',
        'TSMixer': 'https://arxiv.org/abs/2303.06053'
    };
    
    forecastData.model_rankings.forEach((model, index) => {
        const row = document.createElement('tr');
        
        // Performance badge based on MAPE and MASE scores (the two most recommended metrics)
        let performanceBadge = '';
        let badgeClass = '';
        
        // Use the average of MAPE and MASE for performance evaluation
        // MASE < 1 means better than naive forecast, MAPE shows percentage error
        const avgMapeOnly = model.mape; // MAPE is primary metric for ranking
        
        if (avgMapeOnly < 10) {
            performanceBadge = 'Excellent';
            badgeClass = 'bg-green-100 text-green-800';
        } else if (avgMapeOnly < 15) {
            performanceBadge = 'Good';
            badgeClass = 'bg-blue-100 text-blue-800';
        } else if (avgMapeOnly < 25) {
            performanceBadge = 'Fair';
            badgeClass = 'bg-yellow-100 text-yellow-800';
        } else {
            performanceBadge = 'Poor';
            badgeClass = 'bg-red-100 text-red-800';
        }
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                ${index + 1}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${modelLinks[model.model_name] ? 
                    `<a href="${modelLinks[model.model_name]}" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline">${model.model_name}</a>` : 
                    model.model_name
                }
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${model.mape.toFixed(2)}%
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${model.mase !== null && model.mase !== undefined ? model.mase.toFixed(2) : 'N/A'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${model.rmsse !== null && model.rmsse !== undefined ? model.rmsse.toFixed(2) : 'N/A'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${model.mae ? Math.round(model.mae).toLocaleString() : 'N/A'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${badgeClass}">
                    ${performanceBadge}
                </span>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

/**
 * Populate validation against actuals table
 */
function populateValidationTable() {
    const tableBody = document.getElementById('validationTable');
    tableBody.innerHTML = '';
    
    // Get selected model
    const selectedModel = document.getElementById('validationModelSelector').value;
    
    // Get validation data for selected model
    let validationData = forecastData.all_models_validation?.[selectedModel] || [];
    
    if (validationData.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="6" class="px-6 py-4 text-center text-gray-500">
                No validation data available
            </td>
        `;
        tableBody.appendChild(row);
        return;
    }
    
    let totalError = 0;
    let totalPercentError = 0;
    let validationCount = 0; // Only count completed months
    
    // Get current month for comparison
    const currentMonth = new Date().toISOString().substring(0, 7); // YYYY-MM format
    
    validationData.forEach((item, index) => {
        const row = document.createElement('tr');
        
        // Check if this is the current ongoing month using the backend flag
        const isCurrentMonth = item.is_current_month === true;
        
        // Only include completed months in accuracy calculations
        if (!isCurrentMonth) {
            totalError += Math.abs(item.error); // Use absolute value for average error calculation
            totalPercentError += item.percent_error;
            validationCount++;
        }
        
        // Performance badge based on percent error (similar to model rankings)
        let performanceBadge = '';
        let badgeClass = '';
        
        if (!isCurrentMonth) {
            const absPercentError = Math.abs(item.percent_error);
            if (absPercentError === 0) {
                performanceBadge = 'Perfect';
                badgeClass = 'bg-purple-100 text-purple-800';
            } else if (absPercentError < 10) {
                performanceBadge = 'Excellent';
                badgeClass = 'bg-green-100 text-green-800';
            } else if (absPercentError < 15) {
                performanceBadge = 'Good';
                badgeClass = 'bg-blue-100 text-blue-800';
            } else if (absPercentError < 25) {
                performanceBadge = 'Fair';
                badgeClass = 'bg-yellow-100 text-yellow-800';
            } else {
                performanceBadge = 'Poor';
                badgeClass = 'bg-red-100 text-red-800';
            }
        }
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${formatMonth(item.date)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                ${item.actual.toLocaleString()}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${Math.round(item.predicted).toLocaleString()}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 ${isCurrentMonth ? 'text-gray-400' : ''}">
                ${isCurrentMonth ? '-' : (item.error > 0 ? '+' + Math.round(item.error).toLocaleString() : Math.round(item.error).toLocaleString())}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 ${isCurrentMonth ? 'text-gray-400' : ''}">
                ${isCurrentMonth ? '-' : item.percent_error.toFixed(2) + '%'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                ${isCurrentMonth ? '<span class="text-gray-400">-</span>' : `<span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${badgeClass}">${performanceBadge}</span>`}
            </td>
        `;
        
        row.className = 'validation-row';
        if (isCurrentMonth) {
            row.className += ' bg-blue-50'; // Highlight current month row
        }
        
        tableBody.appendChild(row);
    });
    
    // Update summary statistics (excluding current month)
    if (validationCount > 0) {
        const avgError = Math.round(totalError / validationCount).toLocaleString();
        
        // Use MAPE from model rankings instead of calculating manually to ensure consistency
        const selectedModelRanking = forecastData.model_rankings.find(model => model.model_name === selectedModel);
        const mapeValue = selectedModelRanking ? selectedModelRanking.mape.toFixed(2) : (totalPercentError / validationCount).toFixed(2);
        
        // Update both card displays and bottom summary
        document.getElementById('avgErrorCard').textContent = avgError;
        document.getElementById('avgPercentErrorCard').textContent = `${mapeValue}%`;
        
        document.getElementById('avgError').textContent = avgError;
        document.getElementById('avgPercentError').textContent = `${mapeValue}%`;
    } else {
        // No completed months to calculate from
        document.getElementById('avgErrorCard').textContent = '-';
        document.getElementById('avgPercentErrorCard').textContent = '-';
        
        document.getElementById('avgError').textContent = '-';
        document.getElementById('avgPercentError').textContent = '-';
    }
}
function updateDataPeriodInfo() {
    const summary = forecastData.summary;
    
    // Format dates nicely
    const formatDate = (dateStr) => {
        // Parse the date string explicitly to avoid timezone issues
        const [year, month, day] = dateStr.split('-');
        const date = new Date(year, month - 1, day);
        return date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
    };
    
    // Update historical data period as a formatted range
    document.getElementById('historicalPeriod').textContent = 
        `${formatDate(summary.data_period.start)} to ${formatDate(summary.data_period.end)}`;
    
    document.getElementById('forecastPeriod').textContent = 
        `${formatDate(summary.forecast_period.start)} to ${formatDate(summary.forecast_period.end)}`;
}

/**
 * Update current year forecast card using backend pre-calculated totals
 */
function updateCurrentYearForecast() {
    // Use backend pre-calculated totals (no JavaScript calculations needed)
    const backendTotals = forecastData.yearly_forecast_totals || {};
    const yearlyTotal = backendTotals.best_model_total || 0;
    const bestModelName = backendTotals.best_model_name || 'Unknown';
    
    // Update the display
    document.getElementById('currentYearForecast').textContent = yearlyTotal.toLocaleString();
    
    // Update the description to show best model (not average)
    document.getElementById('forecastDescription').textContent = 
        `Total CVEs: Published + Forecasted (${bestModelName} - best model)`;
}

/**
 * SINGLE SOURCE OF TRUTH: Calculate yearly forecast total for any model selection
 * This function is the authoritative calculation used by all UI components
 * @param {string} selectedModel - Either "all" for average of all models, or specific model name
 * @returns {number} - Rounded integer total of forecasted CVEs for the year
 */
function calculateYearlyForecastTotal(selectedModel) {
    if (!forecastData || !forecastData.historical_data) return 0;
    
    let total = 0;
    
    // Step 1: Sum all historical data cve_count (completed months)
    forecastData.historical_data.forEach(item => {
        // Only include 2025 data, not decades of historical data
        if (item.date && item.date.startsWith('2025')) {
            total += item.cve_count;
        }
    });
    
    // Step 2: Add current month actual if available
    if (forecastData.current_month_actual) {
        total += forecastData.current_month_actual.cve_count;
    }
    
    // Step 3: Add forecast totals based on selected model
    if (selectedModel === 'all') {
        // Calculate average of all model forecasts for each remaining month
        const allModelNames = Object.keys(forecastData.forecasts);
        if (allModelNames.length === 0) return Math.round(total);
        
        // Group forecasts by month to calculate averages
        const monthlyAverages = {};
        
        allModelNames.forEach(modelName => {
            const modelForecasts = forecastData.forecasts[modelName];
            if (modelForecasts) {
                modelForecasts.forEach(forecast => {
                    // Only include future months to avoid double-counting current month
                    const [year, month] = forecast.date.split('-').map(Number);
                    const currentMonth = new Date().getMonth() + 1;
                    
                    if (month > currentMonth || year > 2025) {
                        if (!monthlyAverages[forecast.date]) {
                            monthlyAverages[forecast.date] = [];
                        }
                        monthlyAverages[forecast.date].push(forecast.cve_count);
                    }
                });
            }
        });
        
        // Calculate average for each month and add to total
        Object.keys(monthlyAverages).forEach(monthKey => {
            const monthValues = monthlyAverages[monthKey];
            const monthAverage = monthValues.reduce((sum, val) => sum + val, 0) / monthValues.length;
            total += monthAverage;
        });
        
    } else {
        // Use specific model forecasts
        if (forecastData.forecasts[selectedModel]) {
            const modelForecasts = forecastData.forecasts[selectedModel];
            const currentMonth = new Date().getMonth() + 1;
            
            modelForecasts.forEach(forecast => {
                // Only include future months to avoid double-counting current month
                const [year, month] = forecast.date.split('-').map(Number);
                if (month > currentMonth || year > 2025) {
                    total += forecast.cve_count;
                }
            });
        }
    }
    
    return Math.round(total);
}

/**
 * Update time range selector with dynamic year values
 */
function updateTimeRangeSelector() {
    const currentYear = new Date().getFullYear();
    const nextYear = currentYear + 1;
    
    const timeRangeSelector = document.getElementById('timeRange');
    
    // Handle case where dropdown was removed during simplification
    if (!timeRangeSelector) {
        console.log('✅ Time range selector not found - using simplified interface with main chart only');
        return;
    }
    
    // Update the text of the year options
    const thisYearOption = timeRangeSelector.querySelector('option[value="this_year"]');
    const nextYearOption = timeRangeSelector.querySelector('option[value="next_year"]');
    
    if (thisYearOption) {
        thisYearOption.textContent = `This Year (${currentYear})`;
    }
    
    if (nextYearOption) {
        nextYearOption.textContent = `Next Year (${nextYear})`;
    }
}

/**
 * Create the main forecast chart
 */
function createChart() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Prepare initial data
    const chartData = prepareChartData();
    
    try {
        chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'CVE Publications: Monthly Historical Data and Forecasts'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: function(context) {
                                if (context.length > 0) {
                                    const point = context[0];
                                    const date = new Date(point.parsed.x);
                                    const year = date.getUTCFullYear();
                                    const month = date.getUTCMonth(); // 0-indexed
                                    
                                    // Format date properly for tooltip title
                                    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                                    const formattedDate = `${monthNames[month]} ${year}`;
                                    
                                    const monthForCompare = month + 1;
                                    const dateStrCompare = `${year}-${String(monthForCompare).padStart(2, '0')}`;

                                    if (forecastData.current_month_actual &&
                                        dateStrCompare === forecastData.current_month_actual.date) {
                                        return `${formattedDate} - Current Month (${forecastData.current_month_actual.progress_percentage.toFixed(2)}% Complete)`;
                                    }
                                    return formattedDate;
                                }
                                return '';
                            },
                            label: function(context) {
                                const point = context;
                                const value = formatNumber(point.parsed.y);
                                
                                // Extract YYYY-MM for comparison using UTC to avoid timezone issues
                                const date = new Date(point.parsed.x);
                                const year = date.getUTCFullYear();
                                const month = String(date.getUTCMonth() + 1).padStart(2, '0');
                                const dateStr = `${year}-${month}`;
                                
                                // Check if this is the current month and add special information
                                if (forecastData.current_month_actual && 
                                    dateStr === forecastData.current_month_actual.date &&
                                    point.datasetIndex === 0) { // Only for the historical data series
                                    const progress = forecastData.current_month_actual;
                                    return [
                                        `${point.dataset.label}: ${value} CVEs`,
                                        `Days elapsed: ${progress.days_elapsed} of ${progress.total_days}`,
                                        `Month progress: ${progress.progress_percentage.toFixed(2)}%`
                                    ];
                                }
                                
                                return `${point.dataset.label}: ${value}`;
                            },
                            afterLabel: function(context) {
                                const point = context;
                                
                                // Extract YYYY-MM for comparison using UTC to avoid timezone issues
                                const date = new Date(point.parsed.x);
                                const year = date.getUTCFullYear();
                                const month = String(date.getUTCMonth() + 1).padStart(2, '0');
                                const dateStr = `${year}-${month}`;
                                
                                // Add extra context for current month
                                if (forecastData.current_month_actual && 
                                    dateStr === forecastData.current_month_actual.date &&
                                    point.datasetIndex === 0) {
                                    return '📊 Partial month data';
                                }
                                
                                return null;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        min: '2025-01-01',
                        max: '2026-01-01',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM yyyy'
                            },
                            parser: false, // Let Chart.js handle parsing
                            round: 'month',
                            tooltipFormat: 'MMM yyyy'
                        },
                        adapters: {
                            date: {}
                        },
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Number of CVEs'
                        },
                        beginAtZero: true
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
        console.log('Chart created successfully');
    } catch (error) {
        console.error('Error creating chart:', error);
        throw error;
    }
}

/**
 * Convert monthly data to cumulative sum
 * Shows cumulative total of CVEs published BEFORE the first day of each month
 */
function calculateCumulativeSum(monthlyData) {
    const result = [];
    let cumulativeSum = 0;
    
    for (let i = 0; i < monthlyData.length; i++) {
        const item = monthlyData[i];
        
        // The data point shows CVEs published BEFORE this month
        result.push({
            date: item.date,
            cve_count: cumulativeSum  // Total before this month
        });
        
        // Add this month's CVEs to the running total for next iteration
        cumulativeSum += item.cve_count;
    }
    
    return result;
}

/**
 * Debug function to verify chart data alignment
 * Logs all datasets with their 12-month structure for verification
 */
function debugChartDataAlignment(datasets, fullYearMonths) {
    console.log('=== CHART DATA ALIGNMENT DEBUG ===');
    console.log('Full year timeline:', fullYearMonths);
    
    let debugOutput = [];
    
    datasets.forEach((dataset, datasetIndex) => {
        debugOutput.push(`\n--- Dataset ${datasetIndex + 1}: ${dataset.label} ---`);
        
        dataset.data.forEach((point, pointIndex) => {
            const month = fullYearMonths[pointIndex];
            const value = point.y;
            const status = value === null ? 'NULL' : value;
            
            debugOutput.push(`Month: ${month}, Model: ${dataset.label}, Value: ${status}`);
        });
    });
    
    const fullDebugText = debugOutput.join('\n');
    console.log(fullDebugText);
    
    // Also log a summary
    console.log('\n=== ALIGNMENT SUMMARY ===');
    console.log(`Total datasets: ${datasets.length}`);
    console.log(`Points per dataset: ${datasets[0]?.data?.length || 0}`);
    console.log(`Expected points per dataset: 13 (Jan 2025 - Jan 2026)`);
    
    // Verify all datasets have same length
    const allSameLength = datasets.every(dataset => dataset.data.length === 13);
    console.log(`All datasets have 13 points: ${allSameLength}`);
    
    if (!allSameLength) {
        console.warn('⚠️ DATA ALIGNMENT ISSUE: Not all datasets have 13 points!');
        datasets.forEach((dataset, index) => {
            console.warn(`Dataset ${index + 1} (${dataset.label}): ${dataset.data.length} points`);
        });
    } else {
        console.log('✅ Data alignment verified: All datasets have consistent 13-month structure (Jan 2025 - Jan 2026)');
    }
    
    console.log('=== END DEBUG ===\n');
}

/**
 * Prepare main chart data using backend pre-calculated cumulative timelines
 * Shows historical data + 5 ML model forecasts + current month actual
 * NO JavaScript calculations needed - all data pre-calculated by Python backend
 */
function prepareChartData() {
    // Simplified: Always show main cumulative chart with all models
    const selectedModel = 'all';
    const timeRange = 'this_year';
    const chartType = 'cumulative';
    
    console.log('Using backend pre-calculated data for model:', selectedModel, 'chart type:', chartType);
    
    // Use backend pre-calculated cumulative timelines (no calculations needed)
    const cumulativeTimelines = forecastData.cumulative_timelines || {};
    
    if (chartType === 'monthly') {
        // For monthly view, we still need the original monthly data structure
        return prepareMonthlyChartData(selectedModel, timeRange);
    }
    
    // CRITICAL FIX: All datasets must span full timeline to prevent Chart.js alignment issues
    const datasets = [];
    const currentMonth = 7; // July 2025
    
    // Create consistent timeline for all datasets (Jan 2025 - Jan 2026)
    const fullTimeline = [];
    for (let month = 1; month <= 12; month++) {
        fullTimeline.push(`2025-${String(month).padStart(2, '0')}`);
    }
    fullTimeline.push('2026-01'); // Add January 2026
    
    // MAIN CHART: Historical + 5 Model Forecasts + Average (all with full timeline)
    // Simplified: Always show main chart with all models
        
        // 1. Historical Data (Solid Blue Line) - Full timeline with nulls for future
        const historicalDataArray = new Array(fullTimeline.length).fill(null);
        
        // CRITICAL FIX: Read directly from backend cumulative_timelines to prevent data alignment bugs
        const referenceTimeline = cumulativeTimelines['DLinear_cumulative'] || [];
        
        fullTimeline.forEach((monthKey, index) => {
            const month = parseInt(monthKey.split('-')[1]);
            const year = parseInt(monthKey.split('-')[0]);
            
            if (year === 2025 && month < currentMonth) {
                // Historical months - read directly from backend cumulative timeline
                const backendItem = referenceTimeline.find(item => item.date === monthKey);
                if (backendItem) {
                    historicalDataArray[index] = backendItem.cumulative_total;
                }
            } else if (year === 2025 && month === currentMonth) {
                // Current month - use pre-calculated cumulative_total from current_month_actual
                if (forecastData.current_month_actual && forecastData.current_month_actual.cumulative_total) {
                    historicalDataArray[index] = forecastData.current_month_actual.cumulative_total;
                }
            }
            // Future months remain null
        });
        
        console.log('✅ Using backend cumulative timeline data directly - no JavaScript recalculation');
        
        console.log('✅ Cumulative chart starts at January 1st, 2025 with value:', historicalDataArray[0]);
        
        const historicalData = historicalDataArray.map((value, index) => ({
            x: fullTimeline[index] + '-01',
            y: value
        }));
        
        // DEBUG: Log chart data structure and alignment
        console.log('🔍 CHART DATA ALIGNMENT DEBUG:');
        console.log('Full timeline:', fullTimeline);
        console.log('Historical data array values (first 8):', historicalDataArray.slice(0, 8));
        console.log('Historical Chart.js data points (first 8):', historicalData.slice(0, 8));
        
        // Debug specific months to identify alignment issue
        console.log('🔍 SPECIFIC MONTH ALIGNMENT:');
        console.log('Timeline[1] (Feb):', fullTimeline[1], '-> Value:', historicalDataArray[1]);
        console.log('Timeline[2] (Mar):', fullTimeline[2], '-> Value:', historicalDataArray[2]);
        console.log('Expected: Feb should be 4314, Mar should be 8027');
        
        datasets.push({
            label: 'Published CVEs (Actual)',
            data: historicalData,
            borderColor: 'rgb(59, 130, 246)', // Solid blue
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 3,
            pointRadius: 3,
            pointHoverRadius: 6,
            fill: false,
            tension: 0.1
        });
        
        // 2. Individual Model Forecast Lines (Dotted/Dashed) - Current + Future Months
        const modelColors = {
            'DLinear': 'rgb(239, 68, 68)',     // Red
            'NHiTS': 'rgb(34, 197, 94)',       // Green  
            'NBEATS': 'rgb(168, 85, 247)',     // Purple
            'KalmanFilter': 'rgb(251, 191, 36)', // Yellow
            'XGBoost': 'rgb(236, 72, 153)'     // Pink
        };
        
        // Get top 5 models from rankings
        const top5Models = forecastData.model_rankings ? forecastData.model_rankings.slice(0, 5) : [];
        
        top5Models.forEach((modelRanking, index) => {
            const modelName = modelRanking.model_name;
            const modelTimeline = cumulativeTimelines[`${modelName}_cumulative`] || [];
            const color = modelColors[modelName] || 'rgb(107, 114, 128)';
            
            // DEBUG: Log individual model timeline data
            console.log(`🔍 MODEL DEBUG - ${modelName}:`);
            console.log(`Timeline length: ${modelTimeline.length}`);
            if (modelTimeline.length > 6) {
                console.log(`July (2025-07): ${modelTimeline[6]?.cumulative_total || 'NOT FOUND'}`);
                console.log(`August (2025-08): ${modelTimeline[7]?.cumulative_total || 'NOT FOUND'}`);
            }
            
            // Create forecast data array spanning full timeline with nulls for past months
            const forecastDataArray = new Array(fullTimeline.length).fill(null);
            
            fullTimeline.forEach((monthKey, timelineIndex) => {
                const month = parseInt(monthKey.split('-')[1]);
                const year = parseInt(monthKey.split('-')[0]);
                
                // Only fill forecast data for current + future months
                if ((year === 2025 && month >= currentMonth) || year === 2026) {
                    const timelineItem = modelTimeline.find(item => item.date === monthKey);
                    if (timelineItem) {
                        forecastDataArray[timelineIndex] = timelineItem.cumulative_total;
                    }
                }
                // Past months remain null
            });
            
            const forecastData = forecastDataArray.map((value, index) => ({
                x: fullTimeline[index] + '-01',
                y: value
            }));
            
            datasets.push({
                label: `${modelName} (Forecast)`,
                data: forecastData,
                borderColor: color,
                backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
                borderWidth: 2,
                borderDash: [5, 5], // Dashed line for forecasts
                pointRadius: 3,
                pointHoverRadius: 6,
                fill: false,
                tension: 0.1,
                spanGaps: false // Don't connect null values
            });
        });
        
        // 3. All Models Average Forecast Line (Dotted) - Full timeline with nulls for past
        const avgTimeline = cumulativeTimelines['all_models_cumulative'] || [];
        const avgForecastDataArray = new Array(fullTimeline.length).fill(null);
        
        fullTimeline.forEach((monthKey, timelineIndex) => {
            const month = parseInt(monthKey.split('-')[1]);
            const year = parseInt(monthKey.split('-')[0]);
            
            // Only fill forecast data for current + future months
            if ((year === 2025 && month >= currentMonth) || year === 2026) {
                const timelineItem = avgTimeline.find(item => item.date === monthKey);
                if (timelineItem) {
                    avgForecastDataArray[timelineIndex] = timelineItem.cumulative_total;
                }
            }
            // Past months remain null
        });
        
        const avgForecastData = avgForecastDataArray.map((value, index) => ({
            x: fullTimeline[index] + '-01',
            y: value
        }));
        
        datasets.push({
            label: 'All Models Average (Forecast)',
            data: avgForecastData,
            borderColor: 'rgb(75, 85, 99)', // Gray
            backgroundColor: 'rgba(75, 85, 99, 0.1)',
            borderWidth: 3,
            borderDash: [2, 2], // Dotted line for average
            pointRadius: 4,
            pointHoverRadius: 7,
            fill: false,
            tension: 0.1,
            spanGaps: false // Don't connect null values
        });
    
    console.log('✅ Main chart with 5 ML models, historical data, and current month actual successfully created');


    
    // Return chart configuration with backend pre-calculated data
    return {
        datasets: datasets,
        options: getChartOptions(chartType)
    };
}

/**
 * Get cumulative total of historical data up to specified month
 */
function getHistoricalCumulativeTotal(monthKey) {
    let cumulativeTotal = 0;
    
    // Sum all historical data up to and including the specified month
    for (const item of forecastData.historical_data) {
        if (item.date <= monthKey && item.date.startsWith('2025')) {
            cumulativeTotal += item.cve_count;
        }
    }
    
    return cumulativeTotal;
}

/**
 * Get Chart.js configuration options for different chart types
 */
function getChartOptions(chartType) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: '#374151',
                borderWidth: 1,
                cornerRadius: 6,
                displayColors: true,
                callbacks: {
                    title: function(tooltipItems) {
                        if (tooltipItems.length > 0) {
                            const date = new Date(tooltipItems[0].parsed.x);
                            const options = { year: 'numeric', month: 'long' };
                            return date.toLocaleDateString('en-US', options);
                        }
                    },
                    label: function(context) {
                        const value = context.parsed.y;
                        const label = context.dataset.label || '';
                        return `${label}: ${value.toLocaleString()} CVEs`;
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 20
                }
            }
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'month',
                    displayFormats: {
                        month: 'MMM yyyy'
                    }
                },
                title: {
                    display: true,
                    text: 'Month'
                },
                min: '2025-01-01',
                max: '2026-01-01'
            },
            y: {
                beginAtZero: chartType === 'cumulative',
                title: {
                    display: true,
                    text: chartType === 'cumulative' ? 'Cumulative CVEs' : 'Monthly CVEs'
                },
                ticks: {
                    callback: function(value) {
                        return value.toLocaleString();
                    }
                }
            }
        }
    };
}

/**
 * Update chart with new data
 */
function updateChart() {
    if (chart) {
        const chartType = document.getElementById('chartType').value;
        
        // Update chart data
        chart.data = prepareChartData();
        
        // Update chart title and y-axis label based on chart type
        const titleText = chartType === 'cumulative' 
            ? 'CVE Publications: Cumulative Growth and Forecasts (Monthly)'
            : 'CVE Publications: Monthly Historical Data and Forecasts';
            
        const yAxisLabel = chartType === 'cumulative'
            ? 'Cumulative Number of CVEs'
            : 'Number of CVEs per Month';
        
        chart.options.plugins.title.text = titleText;
        chart.options.scales.y.title.text = yAxisLabel;
        
        chart.update();
    }
}

/**
 * Utility function to format numbers with two decimal places
 */
function formatNumber(num) {
    // For CVE counts, always show as integers with comma formatting
    if (Number.isInteger(num) || Math.abs(num - Math.round(num)) < 0.01) {
        return Math.round(num).toLocaleString();
    } else {
        // For metrics and percentages, show 2 decimal places
        return parseFloat(num).toFixed(2);
    }
}

/**
 * Utility function to format month strings
 */
function formatMonth(monthString) {
    // Input format: YYYY-MM, output format: MMM YYYY
    // Use UTC to avoid timezone issues
    const [year, month] = monthString.split('-');
    const date = new Date(parseInt(year), parseInt(month) - 1, 1);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
}

/**
 * Utility function to format dates
 */
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
}
