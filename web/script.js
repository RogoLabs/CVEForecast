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
        'TiDE': 'https://arxiv.org/abs/2304.08424',
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
        'LinearRegression': 'https://en.wikipedia.org/wiki/Linear_regression',
        'Theta': 'https://en.wikipedia.org/wiki/Theta_model',
        'FourTheta': 'https://github.com/Nixtla/statsforecast/blob/main/statsforecast/models.py#L1427',
        'Prophet': 'https://facebook.github.io/prophet/',
        'Croston': 'https://en.wikipedia.org/wiki/Croston%27s_method',
        'NaiveSeasonal': 'https://otexts.com/fpp3/simple-methods.html#seasonal-naïve-method',
        'NaiveMean': 'https://otexts.com/fpp3/simple-methods.html#average-method',
        'NaiveEnsemble': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.ensemble_model.html',
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
                ${model.mape !== null && model.mape !== undefined ? model.mape.toFixed(2) + '%' : 'N/A'}
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
    
    // Get ALL validation data for selected model (not just 2025)
    let validationData = forecastData.all_models_validation?.[selectedModel] || [];
    
    if (validationData.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="6" class="px-6 py-4 text-center text-gray-500">
                No validation data available for this model
            </td>
        `;
        tableBody.appendChild(row);
        return;
    }
    
    // Group validation data by year
    const dataByYear = {};
    let totalError = 0;
    let totalPercentError = 0;
    let validationCount = 0; // Only count completed months
    
    validationData.forEach(item => {
        // Calculate error and percent_error if missing from backend data
        if (item.error === undefined || item.error === null) {
            item.error = item.predicted - item.actual;
        }
        if (item.percent_error === undefined || item.percent_error === null) {
            item.percent_error = item.actual !== 0 ? ((item.predicted - item.actual) / item.actual) * 100 : 0;
        }
        
        const year = item.date.substring(0, 4);
        if (!dataByYear[year]) {
            dataByYear[year] = [];
        }
        dataByYear[year].push(item);
        
        // Check if this is the current ongoing month using the backend flag
        const isCurrentMonth = item.is_current_month === true;
        
        // Only include completed months in accuracy calculations
        if (!isCurrentMonth) {
            totalError += Math.abs(item.error);
            totalPercentError += item.percent_error;
            validationCount++;
        }
    });
    
    // Sort years in descending order (most recent first)
    const sortedYears = Object.keys(dataByYear).sort((a, b) => parseInt(b) - parseInt(a));
    
    // Create expandable year sections
    sortedYears.forEach(year => {
        const yearData = dataByYear[year];
        const yearCount = yearData.filter(item => !item.is_current_month).length;
        
        // Calculate yearly totals (only for completed months)
        const completedMonths = yearData.filter(item => !item.is_current_month);
        const yearlyActual = completedMonths.reduce((sum, item) => sum + item.actual, 0);
        const yearlyPredicted = completedMonths.reduce((sum, item) => sum + Math.round(item.predicted), 0);
        const yearlyError = yearlyPredicted - yearlyActual;
        const yearlyPercentError = yearlyActual > 0 ? ((yearlyError / yearlyActual) * 100) : 0;
        
        // Create year header row (expandable)
        const yearHeaderRow = document.createElement('tr');
        yearHeaderRow.className = 'year-header-row bg-gray-50 hover:bg-gray-100 cursor-pointer border-t-2 border-gray-200';
        yearHeaderRow.innerHTML = `
            <td class="px-6 py-3 text-left">
                <div class="flex items-center">
                    <svg class="w-4 h-4 mr-2 transform transition-transform duration-200 year-toggle-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                    </svg>
                    <span class="font-semibold text-gray-700">${year}</span>
                    <span class="ml-2 text-sm text-gray-500">(${yearCount} month${yearCount !== 1 ? 's' : ''})</span>
                </div>
            </td>
            <td class="px-6 py-3 text-sm font-medium text-gray-900">
                ${yearlyActual.toLocaleString()}
            </td>
            <td class="px-6 py-3 text-sm text-gray-900">
                ${yearlyPredicted.toLocaleString()}
            </td>
            <td class="px-6 py-3 text-sm text-gray-900">
                ${yearlyError > 0 ? '+' + yearlyError.toLocaleString() : yearlyError.toLocaleString()}
            </td>
            <td class="px-6 py-3 text-sm text-gray-900">
                ${yearlyPercentError.toFixed(2)}%
            </td>
            <td class="px-6 py-3 text-sm text-gray-500">
                Year Total
            </td>
        `;
        
        // Add click handler for expanding/collapsing
        yearHeaderRow.addEventListener('click', function() {
            const yearRows = tableBody.querySelectorAll(`[data-year="${year}"]`);
            const toggleIcon = this.querySelector('.year-toggle-icon');
            const isExpanded = toggleIcon.classList.contains('rotate-90');
            
            yearRows.forEach(row => {
                row.style.display = isExpanded ? 'none' : 'table-row';
            });
            
            if (isExpanded) {
                toggleIcon.classList.remove('rotate-90');
            } else {
                toggleIcon.classList.add('rotate-90');
            }
        });
        
        tableBody.appendChild(yearHeaderRow);
        
        // Create month rows for this year (initially hidden)
        yearData.forEach(item => {
            const isCurrentMonth = item.is_current_month === true;
            
            // Performance badge based on percent error
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
            
            const monthRow = document.createElement('tr');
            monthRow.className = `month-data-row ${isCurrentMonth ? 'bg-blue-50' : ''}`;
            monthRow.setAttribute('data-year', year);
            monthRow.style.display = 'none'; // Initially hidden
            
            monthRow.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-12">
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
                    ${isCurrentMonth ? '-' : (item.percent_error !== null && item.percent_error !== undefined ? item.percent_error.toFixed(2) + '%' : 'N/A')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    ${isCurrentMonth ? '<span class="text-gray-400">-</span>' : `<span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${badgeClass}">${performanceBadge}</span>`}
                </td>
            `;
            
            tableBody.appendChild(monthRow);
        });
    });
    
    // Update summary statistics (excluding current month)
    if (validationCount > 0) {
        const avgError = Math.round(totalError / validationCount).toLocaleString();
        
        // Use MAPE from model rankings (backend calculation is correct)
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

/**
 * Update data period info
 */
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
 * Update current year forecast card using dynamic forecast data from new_forecast_runs
 */
function updateCurrentYearForecast() {
    // Get best model name from model rankings (first model is best)
    let bestModelName = 'Unknown';
    if (forecastData.model_rankings && forecastData.model_rankings.length > 0) {
        bestModelName = forecastData.model_rankings[0].model_name;
    }
    
    // Get forecast total from new_forecast_runs (dynamic forecasting data)
    let forecastTotal = 0;
    
    // First try the new dynamic forecast data
    if (forecastData.new_forecast_runs && forecastData.new_forecast_runs.yearly_forecast_totals) {
        forecastTotal = forecastData.new_forecast_runs.yearly_forecast_totals[bestModelName] || 0;
    }
    
    // Fallback to old cumulative timeline data if new data not available
    if (forecastTotal === 0) {
        const bestModelCumulative = `${bestModelName}_cumulative`;
        if (forecastData.cumulative_timelines && forecastData.cumulative_timelines[bestModelCumulative]) {
            const timeline = forecastData.cumulative_timelines[bestModelCumulative];
            // Get the Jan 1 2026 endpoint (last entry in timeline)
            const jan2026Entry = timeline.find(entry => entry.date === '2026-01');
            if (jan2026Entry) {
                forecastTotal = jan2026Entry.cumulative_total;
            }
        }
    }
    
    // Update the display with forecast total
    document.getElementById('currentYearForecast').textContent = forecastTotal.toLocaleString();
    
    // Update the description to show best model and data source
    const dataSource = (forecastData.new_forecast_runs && forecastData.new_forecast_runs.yearly_forecast_totals) ? 'Dynamic Forecast' : 'Standard Forecast';
    document.getElementById('forecastDescription').textContent = 
        `Total CVEs: Published + Forecasted (${bestModelName} - ${dataSource})`;
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

                                // If this is the current month, only show the actual (partial) value for the historical data series
                                if (forecastData.current_month_actual && 
                                    dateStr === forecastData.current_month_actual.date) {
                                    if (point.datasetIndex === 0) {
                                        const progress = forecastData.current_month_actual;
                                        return [
                                            `${point.dataset.label}: ${value} CVEs`,
                                            `Days elapsed: ${progress.days_elapsed} of ${progress.total_days}`,
                                            `Month progress: ${progress.progress_percentage.toFixed(2)}%`
                                        ];
                                    } else {
                                        // Suppress forecast data for current month in tooltip
                                        return null;
                                    }
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

                                // If this is the current month, only show extra context for the actual (partial) value
                                if (forecastData.current_month_actual && 
                                    dateStr === forecastData.current_month_actual.date) {
                                    if (point.datasetIndex === 0) {
                                        return '📊 Partial month data';
                                    } else {
                                        // Suppress forecast data for current month in tooltip
                                        return null;
                                    }
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
 * Prepare main chart data using backend pre-calculated cumulative timelines
 * Shows historical data + 5 ML model forecasts + current month actual
 * NO JavaScript calculations needed - all data pre-calculated by Python backend
 */
function prepareChartData() {
    const { cumulative_timelines, model_rankings } = forecastData;
    const datasets = [];

    if (!cumulative_timelines || !model_rankings) {
        console.error("Chart data cannot be prepared: cumulative_timelines is missing.");
        return { datasets: [] };
    }

    // Use the "all_models_cumulative" as the single source of truth for the historical line
    const actualsTimeline = cumulative_timelines.all_models_cumulative || [];

    // Find the last actual data point to anchor the forecasts
    const lastActualPoint = actualsTimeline.length > 0 ? actualsTimeline[actualsTimeline.length - 1] : null;

    // 1. Published CVEs (Actual) - Solid Blue Line
    datasets.push({
        label: 'Published CVEs (Actual)',
        data: actualsTimeline.map(d => ({ x: new Date(d.date), y: d.cumulative_total })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 3,
        pointRadius: 3,
        pointHoverRadius: 6,
        tension: 0.1
    });

    // 2. Forecast Data for Top 5 Models - Dashed/Dotted Lines
    const topModels = model_rankings.slice(0, 5);
    const colors = ['rgb(239, 68, 68)', 'rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(249, 115, 22)', 'rgb(14, 165, 233)'];

    topModels.forEach((model, index) => {
        const modelTimelineKey = `${model.model_name}_cumulative`;
        const modelForecastTimeline = cumulative_timelines[modelTimelineKey];

        if (modelForecastTimeline && lastActualPoint) {
            // THE CRITICAL FIX: Prepend the last actual point to each forecast dataset.
            // This "stitches" the forecast line to the end of the actuals line.
            const stitchedData = [
                { x: new Date(lastActualPoint.date), y: lastActualPoint.cumulative_total },
                ...modelForecastTimeline
                    .filter(d => new Date(d.date) > new Date(lastActualPoint.date))
                    .map(d => ({ x: new Date(d.date), y: d.cumulative_total }))
            ];
            
            datasets.push({
                label: `${model.model_name} (Forecast)`,
                data: stitchedData,
                borderColor: colors[index % colors.length],
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 3,
                pointHoverRadius: 6,
                tension: 0.1
            });
        }
    });

    return { datasets };
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
                        const label = context.dataset.label || '';
                        const value = context.parsed.y;
                        
                        // For all data (including forecasts), show the actual cumulative value
                        return `${label}: ${value.toLocaleString()} CVEs`;
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 20,
                    generateLabels: function(chart) {
                        // Use custom labels instead of Chart.js default behavior
                        return chart.data.datasets.map((dataset, index) => {
                            return {
                                text: dataset.label, // Use our custom label with yearly forecast totals
                                fillStyle: dataset.borderColor,
                                strokeStyle: dataset.borderColor,
                                lineWidth: dataset.borderWidth,
                                hidden: false,
                                index: index
                            };
                        });
                    }
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
