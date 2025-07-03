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
    populateModelRankings();
    updateDataPeriodInfo();
    createChart();
    
    // Add event listeners
    document.getElementById('modelSelector').addEventListener('change', updateChart);
    document.getElementById('timeRange').addEventListener('change', updateChart);
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
    
    // Update best model info
    if (forecastData.model_rankings.length > 0) {
        const bestModel = forecastData.model_rankings[0];
        document.getElementById('bestModel').textContent = bestModel.model_name;
        document.getElementById('bestAccuracy').textContent = `${(bestModel.mape * 100).toFixed(2)}%`;
    }
}

/**
 * Populate model selector dropdown
 */
function populateModelSelector() {
    const selector = document.getElementById('modelSelector');
    
    // Clear existing options except 'All Models'
    selector.innerHTML = '<option value="all">All Models</option>';
    
    // Add options for each model
    Object.keys(forecastData.forecasts).forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        selector.appendChild(option);
    });
}

/**
 * Populate model rankings table
 */
function populateModelRankings() {
    const tableBody = document.getElementById('modelRankingsTable');
    tableBody.innerHTML = '';
    
    forecastData.model_rankings.forEach((model, index) => {
        const row = document.createElement('tr');
        
        // Performance badge
        let performanceBadge = '';
        let badgeClass = '';
        
        if (model.mape < 0.1) {
            performanceBadge = 'Excellent';
            badgeClass = 'bg-green-100 text-green-800';
        } else if (model.mape < 0.2) {
            performanceBadge = 'Good';
            badgeClass = 'bg-blue-100 text-blue-800';
        } else if (model.mape < 0.3) {
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
                ${(model.mape * 100).toFixed(2)}%
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
 * Update data period information
 */
function updateDataPeriodInfo() {
    const summary = forecastData.summary;
    
    document.getElementById('historicalPeriod').textContent = 
        `${summary.data_period.start} to ${summary.data_period.end}`;
    
    document.getElementById('forecastPeriod').textContent = 
        `${summary.forecast_period.start} to ${summary.forecast_period.end}`;
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
 * Aggregate daily data into monthly totals
 */
function aggregateDataByMonth(dailyData) {
    const monthlyData = {};
    
    dailyData.forEach(item => {
        try {
            const date = new Date(item.date);
            
            // Validate date
            if (isNaN(date.getTime())) {
                console.warn('Invalid date found:', item.date);
                return;
            }
            
            // Get the first day of the month as the key
            const monthKey = new Date(date.getFullYear(), date.getMonth(), 1).toISOString().split('T')[0];
            
            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = 0;
            }
            monthlyData[monthKey] += item.cve_count || 0;
        } catch (error) {
            console.warn('Error processing date:', item.date, error);
        }
    });
    
    // Convert back to array format
    return Object.entries(monthlyData).map(([date, count]) => ({
        date: date,
        cve_count: count
    })).sort((a, b) => new Date(a.date) - new Date(b.date));
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
        const cutoffDate = new Date();
        if (timeRange === '6m') {
            cutoffDate.setMonth(cutoffDate.getMonth() - 6);
        } else if (timeRange === '1y') {
            cutoffDate.setFullYear(cutoffDate.getFullYear() - 1);
        }
        
        historicalData = historicalData.filter(item => 
            new Date(item.date) >= cutoffDate
        );
    }
    
    // Aggregate historical data by month
    let monthlyHistoricalData = aggregateDataByMonth(historicalData);
    
    // Apply cumulative sum if requested
    if (chartType === 'cumulative') {
        monthlyHistoricalData = calculateCumulativeSum(monthlyHistoricalData);
    }
    
    console.log('Monthly historical data points:', monthlyHistoricalData.length);
    
    // Prepare datasets
    const datasets = [];
    
    // Historical data
    datasets.push({
        label: chartType === 'cumulative' ? 'Cumulative CVEs' : 'Historical CVEs',
        data: monthlyHistoricalData.map(item => ({
            x: item.date,
            y: item.cve_count
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6
    });
    
    // Forecast data
    if (selectedModel === 'all') {
        // Show all model forecasts
        const colors = [
            'rgb(239, 68, 68)',   // Red
            'rgb(34, 197, 94)',   // Green
            'rgb(168, 85, 247)',  // Purple
            'rgb(245, 158, 11)',  // Amber
            'rgb(236, 72, 153)'   // Pink
        ];
        
        Object.entries(forecastData.forecasts).forEach(([modelName, forecast], index) => {
            const color = colors[index % colors.length];
            
            // Aggregate forecast data by month
            let monthlyForecast = aggregateDataByMonth(forecast);
            
            // For cumulative charts, we need to continue from the last historical value
            if (chartType === 'cumulative') {
                const lastHistoricalValue = monthlyHistoricalData.length > 0 
                    ? monthlyHistoricalData[monthlyHistoricalData.length - 1].cve_count 
                    : 0;
                
                let cumulativeSum = lastHistoricalValue;
                monthlyForecast = monthlyForecast.map(item => {
                    cumulativeSum += item.cve_count;
                    return {
                        date: item.date,
                        cve_count: cumulativeSum
                    };
                });
            }
            
            datasets.push({
                label: `${modelName} ${chartType === 'cumulative' ? 'Cumulative ' : ''}Forecast`,
                data: monthlyForecast.map(item => ({
                    x: item.date,
                    y: item.cve_count
                })),
                borderColor: color,
                backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 3,
                pointHoverRadius: 6
            });
        });
    } else {
        // Show selected model forecast
        if (forecastData.forecasts[selectedModel]) {
            // Aggregate forecast data by month
            let monthlyForecast = aggregateDataByMonth(forecastData.forecasts[selectedModel]);
            
            // For cumulative charts, we need to continue from the last historical value
            if (chartType === 'cumulative') {
                const lastHistoricalValue = monthlyHistoricalData.length > 0 
                    ? monthlyHistoricalData[monthlyHistoricalData.length - 1].cve_count 
                    : 0;
                
                let cumulativeSum = lastHistoricalValue;
                monthlyForecast = monthlyForecast.map(item => {
                    cumulativeSum += item.cve_count;
                    return {
                        date: item.date,
                        cve_count: cumulativeSum
                    };
                });
            }
            
            datasets.push({
                label: `${selectedModel} ${chartType === 'cumulative' ? 'Cumulative ' : ''}Forecast`,
                data: monthlyForecast.map(item => ({
                    x: item.date,
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
            ? 'CVE Publications: Cumulative Growth and Forecasts'
            : 'CVE Publications: Monthly Historical Data and Forecasts';
            
        const yAxisLabel = chartType === 'cumulative'
            ? 'Cumulative Number of CVEs'
            : 'Number of CVEs';
        
        chart.options.plugins.title.text = titleText;
        chart.options.scales.y.title.text = yAxisLabel;
        
        chart.update();
    }
}

/**
 * Utility function to format numbers
 */
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

/**
 * Utility function to format dates
 */
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
}
