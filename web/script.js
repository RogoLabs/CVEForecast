/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions
 */

let forecastData = null;
let chart = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadForecastData();
});

/**
 * Load forecast data from JSON file
 */
async function loadForecastData() {
    console.log('Starting to load forecast data...');
    try {
        const response = await fetch('data.json');
        console.log('Fetch response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        forecastData = await response.json();
        console.log('Forecast data loaded successfully:', Object.keys(forecastData));
        
        // Hide loading state and show dashboard
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');
        
        // Initialize the dashboard
        initializeDashboard();
        
    } catch (error) {
        console.error('Error loading forecast data:', error);
        
        // Show error state
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('errorState').classList.remove('hidden');
    }
}

/**
 * Initialize dashboard components
 */
function initializeDashboard() {
    updateSummaryCards();
    populateModelSelector();
    populateValidationModelSelector();
    populateModelRankings();
    populateValidationTable();
    updateDataPeriodInfo();
    updateTimeRangeSelector();
    createChart();
    
    // Add event listeners
    document.getElementById('modelSelector').addEventListener('change', updateChart);
    document.getElementById('timeRange').addEventListener('change', updateChart);
    document.getElementById('validationModelSelector').addEventListener('change', populateValidationTable);
    if (document.getElementById('chartType')) {
        document.getElementById('chartType').addEventListener('change', updateChart);
    }
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
 */
function populateModelSelector() {
    const selector = document.getElementById('modelSelector');
    
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
                ${model.model_name}
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
 * Calculate and update current year cumulative forecast
 */
function updateCurrentYearForecast() {
    const currentYear = new Date().getFullYear();
    let currentYearTotal = 0;
    
    // Sum up actual CVEs for completed months this year
    if (forecastData.historical_data) {
        forecastData.historical_data.forEach(item => {
            const itemYear = parseInt(item.date.split('-')[0]);
            if (itemYear === currentYear) {
                currentYearTotal += item.cve_count;
            }
        });
    }
    
    // Add current month progress if available
    if (forecastData.current_month_progress) {
        currentYearTotal += forecastData.current_month_progress.cve_count;
    }
    
    // Add forecasts for remaining months this year using the best model
    let bestModelName = "Unknown";
    if (forecastData.model_rankings.length > 0) {
        bestModelName = forecastData.model_rankings[0].model_name;
        const bestModelForecasts = forecastData.forecasts[bestModelName];
        
        if (bestModelForecasts) {
            bestModelForecasts.forEach(forecast => {
                const forecastYear = parseInt(forecast.date.split('-')[0]);
                const forecastMonth = parseInt(forecast.date.split('-')[1]);
                const currentMonth = new Date().getMonth() + 1; // getMonth() is 0-based
                
                // Only include forecasts for remaining months of current year
                if (forecastYear === currentYear && forecastMonth > currentMonth) {
                    currentYearTotal += forecast.cve_count;
                }
            });
        }
    }
    
    // Update the display - round to nearest integer for CVE counts
    document.getElementById('currentYearForecast').textContent = 
        Math.round(currentYearTotal).toLocaleString();
    
    // Update the description to show which model is being used
    document.getElementById('forecastDescription').textContent = 
        `2025 total: actual (Jan-Jul) + forecast (Aug-Dec)`;
}

/**
 * Update time range selector with dynamic year values
 */
function updateTimeRangeSelector() {
    const currentYear = new Date().getFullYear();
    const nextYear = currentYear + 1;
    
    const timeRangeSelector = document.getElementById('timeRange');
    
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
                                const point = context[0];
                                
                                // Use Chart.js's built-in time formatting
                                const chart = this.chart;
                                const scale = chart.scales.x;
                                const timestamp = point.parsed.x;
                                
                                // Format the timestamp using the scale's time adapter
                                const formattedDate = scale._adapter.format(timestamp, 'MMM yyyy');
                                
                                console.log('Tooltip - timestamp:', timestamp, 'formatted:', formattedDate);
                                
                                // Extract YYYY-MM for comparison
                                const date = new Date(timestamp);
                                const year = date.getFullYear();
                                const month = String(date.getMonth() + 1).padStart(2, '0');
                                const dateStr = `${year}-${month}`;
                                
                                // Check if this is the current month
                                if (forecastData.current_month_progress && 
                                    dateStr === forecastData.current_month_progress.date) {
                                    return `${formattedDate} (Current Month - ${forecastData.current_month_progress.progress_percentage.toFixed(2)}% Complete)`;
                                }
                                
                                return formattedDate;
                            },
                            label: function(context) {
                                const point = context;
                                const value = formatNumber(point.parsed.y);
                                
                                // Extract YYYY-MM for comparison
                                const date = new Date(point.parsed.x);
                                const year = date.getFullYear();
                                const month = String(date.getMonth() + 1).padStart(2, '0');
                                const dateStr = `${year}-${month}`;
                                
                                // Check if this is the current month and add special information
                                if (forecastData.current_month_progress && 
                                    dateStr === forecastData.current_month_progress.date &&
                                    point.datasetIndex === 0) { // Only for the historical data series
                                    const progress = forecastData.current_month_progress;
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
                                
                                // Extract YYYY-MM for comparison
                                const date = new Date(point.parsed.x);
                                const year = date.getFullYear();
                                const month = String(date.getMonth() + 1).padStart(2, '0');
                                const dateStr = `${year}-${month}`;
                                
                                // Add extra context for current month
                                if (forecastData.current_month_progress && 
                                    dateStr === forecastData.current_month_progress.date &&
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
 */
function calculateCumulativeSum(monthlyData) {
    let cumulativeSum = 0;
    return monthlyData.map(item => {
        cumulativeSum += item.cve_count;
        return {
            date: item.date,
            cve_count: cumulativeSum
        };
    });
}

/**
 * Prepare chart data based on current selections
 */
function prepareChartData() {
    const selectedModel = document.getElementById('modelSelector').value;
    const timeRange = document.getElementById('timeRange').value;
    const chartType = document.getElementById('chartType').value;
    
    console.log('Preparing chart data for model:', selectedModel, 'time range:', timeRange, 'chart type:', chartType);
    
    // Filter historical data based on time range
    let historicalData = [...forecastData.historical_data];
    
    if (timeRange !== 'all') {
        if (timeRange === 'this_year') {
            // Show all data for the current calendar year (2025)
            const currentYear = new Date().getFullYear();
            historicalData = historicalData.filter(item => {
                const itemYear = parseInt(item.date.split('-')[0]);
                return itemYear === currentYear;
            });
        } else if (timeRange === 'next_year') {
            // Show forecast data for next calendar year (2026)
            const nextYear = new Date().getFullYear() + 1;
            historicalData = historicalData.filter(item => {
                const itemYear = parseInt(item.date.split('-')[0]);
                return itemYear === nextYear;
            });
        }
    }
    
    // Apply cumulative sum if requested (data is already monthly)
    let processedHistoricalData = historicalData;
    if (chartType === 'cumulative') {
        processedHistoricalData = calculateCumulativeSum(historicalData);
    }
    
    console.log('Processed historical data points:', processedHistoricalData.length);
    
    // Prepare datasets
    const datasets = [];
    
    // Historical data
    datasets.push({
        label: chartType === 'cumulative' ? 'Cumulative CVEs' : 'Monthly CVEs',
        data: processedHistoricalData.map(item => ({
            x: item.date + '-01', // Use ISO date string format
            y: item.cve_count
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6
    });
    
    // Current month progress data (if available and relevant for time range)
    if (forecastData.current_month_progress && (timeRange === 'all' || timeRange === 'this_year')) {
        const currentData = forecastData.current_month_progress;
        let currentValue = currentData.cve_count;
        
        // For cumulative charts, add to the last historical value
        if (chartType === 'cumulative' && processedHistoricalData.length > 0) {
            currentValue += processedHistoricalData[processedHistoricalData.length - 1].cve_count;
        }
        
        // Add current month as extension of historical data (blue line)
        const extendedHistoricalData = [...processedHistoricalData.map(item => ({
            x: item.date + '-01',
            y: item.cve_count
        })), {
            x: currentData.date + '-01',
            y: currentValue
        }];
        
        // Update the historical dataset to include current month
        datasets[0].data = extendedHistoricalData;
        datasets[0].label = chartType === 'cumulative' ? 'Cumulative CVEs' : 'Monthly CVEs';
    }
    
    // Forecast data
    if (selectedModel === 'all') {
        // Show only top 5 model forecasts for performance
        const colors = [
            'rgb(239, 68, 68)',   // Red
            'rgb(34, 197, 94)',   // Green
            'rgb(168, 85, 247)',  // Purple
            'rgb(245, 158, 11)',  // Amber
            'rgb(236, 72, 153)'   // Pink
        ];
        
        // Get top 5 models from rankings for chart display
        const top5Models = forecastData.model_rankings ? forecastData.model_rankings.slice(0, 5) : [];
        
        top5Models.forEach((modelRanking, index) => {
            const modelName = modelRanking.model_name;
            const forecast = forecastData.forecasts[modelName];
            
            if (forecast) {
                const color = colors[index % colors.length];
                
                // Filter forecast data based on time range
                let filteredForecast = forecast;
                if (timeRange === 'this_year') {
                    const currentYear = new Date().getFullYear();
                    filteredForecast = forecast.filter(item => {
                        const itemYear = parseInt(item.date.split('-')[0]);
                        return itemYear === currentYear;
                    });
                } else if (timeRange === 'next_year') {
                    const nextYear = new Date().getFullYear() + 1;
                    filteredForecast = forecast.filter(item => {
                        const itemYear = parseInt(item.date.split('-')[0]);
                        return itemYear === nextYear;
                    });
                }
                
                // Process forecast data
                let processedForecast = filteredForecast;
                
                // For cumulative charts, we need to continue from the last historical value
                if (chartType === 'cumulative') {
                    const lastHistoricalValue = processedHistoricalData.length > 0 
                        ? processedHistoricalData[processedHistoricalData.length - 1].cve_count 
                        : 0;
                    
                    let cumulativeSum = lastHistoricalValue;
                    processedForecast = filteredForecast.map(item => {
                        cumulativeSum += item.cve_count;
                        return {
                            date: item.date,
                            cve_count: cumulativeSum
                        };
                    });
                }
                
                datasets.push({
                    label: modelName,
                    data: processedForecast.map(item => ({
                        x: item.date + '-01', // Use ISO date string format
                        y: item.cve_count
                    })),
                    borderColor: color,
                    backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 3,
                    pointHoverRadius: 6
                });
            }
        });
    } else {
        // Show selected model forecast
        if (forecastData.forecasts[selectedModel]) {
            // Filter forecast data based on time range
            let filteredForecast = forecastData.forecasts[selectedModel];
            if (timeRange === 'this_year') {
                const currentYear = new Date().getFullYear();
                filteredForecast = forecastData.forecasts[selectedModel].filter(item => {
                    const itemYear = parseInt(item.date.split('-')[0]);
                    return itemYear === currentYear;
                });
            } else if (timeRange === 'next_year') {
                const nextYear = new Date().getFullYear() + 1;
                filteredForecast = forecastData.forecasts[selectedModel].filter(item => {
                    const itemYear = parseInt(item.date.split('-')[0]);
                    return itemYear === nextYear;
                });
            }
            
            // Process forecast data
            let processedForecast = filteredForecast;
            
            // For cumulative charts, we need to continue from the last historical value
            if (chartType === 'cumulative') {
                const lastHistoricalValue = processedHistoricalData.length > 0 
                    ? processedHistoricalData[processedHistoricalData.length - 1].cve_count 
                    : 0;
                
                let cumulativeSum = lastHistoricalValue;
                processedForecast = filteredForecast.map(item => {
                    cumulativeSum += item.cve_count;
                    return {
                        date: item.date,
                        cve_count: cumulativeSum
                    };
                });
            }
            
            datasets.push({
                label: selectedModel,
                data: processedForecast.map(item => ({
                    x: item.date + '-01', // Use ISO date string format
                    y: item.cve_count
                })),
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 3,
                pointHoverRadius: 6
            });
        }
    }
    
    return {
        datasets: datasets
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
