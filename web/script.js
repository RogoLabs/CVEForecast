/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions.
 * This version is designed as a "dumb client" that only renders
 * pre-calculated data from data.json.
 */

// Global state variables
let forecastData = null;
let chartInstance = null;

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', loadForecastData);

/**
 * Loads forecast data from the data.json file and initializes the dashboard.
 */
async function loadForecastData() {
    console.log('🔄 Loading forecast data...');
    try {
        const response = await fetch('data.json?v=' + new Date().getTime()); // Cache-busting
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        forecastData = await response.json();
        console.log('✅ Forecast data loaded successfully.');

        // Hide loading state and show dashboard
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');

        // Initialize all dashboard components
        initializeDashboard();
        console.log('✅ Dashboard initialized successfully!');

    } catch (error) {
        console.error('❌ Error loading or parsing forecast data:', error);
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('errorState').classList.remove('hidden');
    }
}

/**
 * Initializes all dashboard components with the loaded data.
 */
function initializeDashboard() {
    updateSummaryCards();
    populateModelSelector(); // Note: This dropdown is for the validation table now
    populateValidationModelSelector();
    populateModelRankings();
    populateValidationTable(); // Initially populate with the best model's data
    updateDataPeriodInfo();
    createOrUpdateChart(); // Create the chart for the first time

    // Attach event listener for the validation model selector
    const validationModelSelector = document.getElementById('validationModelSelector');
    if (validationModelSelector) {
        validationModelSelector.addEventListener('change', populateValidationTable);
    }
}

/**
 * Updates the summary cards at the top of the dashboard.
 */
function updateSummaryCards() {
    if (!forecastData) return;

    // Last Updated Timestamp
    document.getElementById('lastUpdated').textContent =
        `Last Updated: ${new Date(forecastData.generated_at).toLocaleString()}`;

    // Use backend-calculated yearly forecast total for the best model
    const bestModelName = forecastData.model_rankings?.[0]?.model_name || 'N/A';
    const yearlyTotals = forecastData.yearly_forecast_totals || {};
    const bestModelTotal = yearlyTotals[bestModelName] || 0;

    document.getElementById('currentYearForecast').textContent = bestModelTotal.toLocaleString();
    document.getElementById('forecastDescription').textContent = `Total CVEs: Published + Forecasted (${bestModelName} - Best Model)`;

    // Best Model Name and Accuracy (MAPE)
    if (forecastData.model_rankings?.length > 0) {
        const bestModel = forecastData.model_rankings[0];
        document.getElementById('bestModel').textContent = bestModel.model_name;
        document.getElementById('bestAccuracy').textContent = `${(bestModel.mape || 0).toFixed(2)}%`;
    }

    // Total Historical CVEs
    document.getElementById('totalCVEs').textContent =
        (forecastData.summary?.total_historical_cves || 0).toLocaleString();
}

/**
 * Populates the model selector dropdowns.
 */
function populateModelSelector() {
    const selector = document.getElementById('validationModelSelector');
    selector.innerHTML = '';
    forecastData.model_rankings?.forEach(model => {
        const option = document.createElement('option');
        option.value = model.model_name;
        option.textContent = model.model_name;
        selector.appendChild(option);
    });
}

function populateValidationModelSelector() {
    populateModelSelector(); // Re-use the same logic
}

/**
 * Populates the model performance rankings table.
 */
function populateModelRankings() {
    const tableBody = document.getElementById('modelRankingsTable');
    tableBody.innerHTML = '';

    forecastData.model_rankings?.forEach((model, index) => {
        const row = document.createElement('tr');
        const mape = model.mape || 0;
        let badgeClass = 'bg-red-100 text-red-800';
        let performanceBadge = 'Poor';

        if (mape < 10) {
            badgeClass = 'bg-green-100 text-green-800';
            performanceBadge = 'Excellent';
        } else if (mape < 15) {
            badgeClass = 'bg-blue-100 text-blue-800';
            performanceBadge = 'Good';
        } else if (mape < 25) {
            badgeClass = 'bg-yellow-100 text-yellow-800';
            performanceBadge = 'Fair';
        }

        row.innerHTML = `
            <td class="px-6 py-4 text-sm font-medium text-gray-900">${index + 1}</td>
            <td class="px-6 py-4 text-sm text-gray-900">${model.model_name}</td>
            <td class="px-6 py-4 text-sm text-gray-900">${mape.toFixed(2)}%</td>
            <td class="px-6 py-4 text-sm text-gray-900">${(model.mase || 0).toFixed(2)}</td>
            <td class="px-6 py-4 text-sm text-gray-900">${(model.rmsse || 0).toFixed(2)}</td>
            <td class="px-6 py-4 text-sm text-gray-900">${Math.round(model.mae || 0).toLocaleString()}</td>
            <td class="px-6 py-4"><span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${badgeClass}">${performanceBadge}</span></td>
        `;
        tableBody.appendChild(row);
    });
}

/**
 * Populates the validation table with data for the selected model.
 */
function populateValidationTable() {
    const tableBody = document.getElementById('validationTable');
    tableBody.innerHTML = '';
    const selectedModel = document.getElementById('validationModelSelector').value;
    const validationData = forecastData.all_models_validation?.[selectedModel] || [];

    if (validationData.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" class="text-center py-4">No validation data available.</td></tr>`;
        return;
    }
    
    // Sort data by date descending to show most recent first
    validationData.sort((a, b) => new Date(b.date) - new Date(a.date));

    validationData.forEach(item => {
        const row = document.createElement('tr');
        row.className = 'validation-row';
        const isCurrent = item.is_current_month;
        const error = isCurrent ? 'N/A' : (item.error > 0 ? `+${Math.round(item.error).toLocaleString()}` : Math.round(item.error).toLocaleString());
        const percentError = isCurrent ? 'N/A' : `${item.percent_error.toFixed(2)}%`;

        row.innerHTML = `
            <td class="px-6 py-3 text-sm text-gray-900">${item.date} ${isCurrent ? '(Current)' : ''}</td>
            <td class="px-6 py-3 text-sm text-gray-900">${Math.round(item.actual).toLocaleString()}</td>
            <td class="px-6 py-3 text-sm text-gray-900">${Math.round(item.predicted).toLocaleString()}</td>
            <td class="px-6 py-3 text-sm text-gray-900">${error}</td>
            <td class="px-6 py-3 text-sm text-gray-900">${percentError}</td>
            <td class="px-6 py-3 text-sm text-gray-900">-</td>
        `;
        tableBody.appendChild(row);
    });
}

/**
 * Updates the data period information cards.
 */
function updateDataPeriodInfo() {
    const summary = forecastData.summary;
    if (!summary) return;

    const formatDate = (dateStr) => new Date(dateStr).toLocaleDateString('en-US', { timeZone: 'UTC', year: 'numeric', month: 'long', day: 'numeric' });

    document.getElementById('historicalPeriod').textContent =
        `${formatDate(summary.data_period.start)} to ${formatDate(summary.data_period.end)}`;
    document.getElementById('forecastPeriod').textContent =
        `${formatDate(summary.forecast_period.start)} to ${formatDate(summary.forecast_period.end)}`;
}

/**
 * Creates or updates the main forecast chart.
 */
function createOrUpdateChart() {
    const chartData = prepareChartData();
    const chartOptions = getChartOptions();

    if (chartInstance) {
        chartInstance.data = chartData;
        chartInstance.options = chartOptions;
        chartInstance.update();
    } else {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: chartOptions,
        });
    }
}

/**
 * Prepares the datasets for the chart.
 * This simplified version ONLY plots the pre-calculated actual cumulative data
 * to ensure it is rendered correctly before adding forecasts back.
 */
function prepareChartData() {
    const { current_year_actual_cumulative } = forecastData;
    const datasets = [];

    if (!current_year_actual_cumulative) {
        console.error("The required 'current_year_actual_cumulative' data is missing from data.json.");
        return { datasets: [] };
    }

    // Create a single dataset for the "Published CVEs (Actual)" line
    // Ensure data is unique by date to prevent duplicate points
    const uniqueActuals = current_year_actual_cumulative.filter((value, index, self) =>
        index === self.findIndex((t) => (
            t.date === value.date
        ))
    );

    const actualsData = uniqueActuals.map(d => ({
        x: new Date(`${d.date}-01T00:00:00Z`), // Ensure correct date parsing
        y: d.cumulative_total
    }));

    datasets.push({
        label: 'Published CVEs (Actual)',
        data: actualsData,
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 3,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        tension: 0.1,
        fill: false,
    });
    
    // Temporarily disable all forecast lines to isolate and fix the actuals line
    console.log("Chart prepared with ONLY actual data. Forecasts are temporarily disabled.");

    return { datasets };
}

/**
 * Returns the configuration options for the chart.
 */
/**
 * Returns the configuration options for the chart.
 */
function getChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } },
            tooltip: {
                mode: 'nearest',
                intersect: true,
                callbacks: {
                    title: (tooltipItems) => {
                        if (!tooltipItems.length) return '';
                        const pointDate = new Date(tooltipItems[0].parsed.x);
                        return pointDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' });
                    },
                    label: (context) => {
                        const label = context.dataset.label || '';
                        const cumulativeTotal = context.parsed.y;

                        return `${label}: ${cumulativeTotal.toLocaleString()}`;
                    },
                },
            },
        },
        scales: {
            x: {
                type: 'time',
                time: { unit: 'month', tooltipFormat: 'MMM yyyy' },
                title: { display: true, text: 'Month' },
                min: '2025-01-01',
                max: '2026-01-01',
            },
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Number of CVEs' },
                ticks: { callback: (value) => value.toLocaleString() },
            },
        },
    };
}