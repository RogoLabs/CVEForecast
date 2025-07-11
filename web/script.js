/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions.
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

        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');

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
    populateModelSelector();
    populateModelRankings();
    populateForecastVsPublishedTable(); // Initially populate with the best model
    updateDataPeriodInfo();
    createOrUpdateChart();

    const validationModelSelector = document.getElementById('validationModelSelector');
    if (validationModelSelector) {
        validationModelSelector.addEventListener('change', populateForecastVsPublishedTable);
    }
}

/**
 * Updates the summary cards at the top of the dashboard.
 */
function updateSummaryCards() {
    if (!forecastData) return;

    document.getElementById('lastUpdated').textContent = `Last Updated: ${new Date(forecastData.generated_at).toLocaleString()}`;

    const bestModelName = forecastData.model_rankings?.[0]?.model_name || 'N/A';
    const yearlyTotals = forecastData.yearly_forecast_totals || {};
    const bestModelTotal = yearlyTotals[bestModelName] || 0;

    document.getElementById('currentYearForecast').textContent = bestModelTotal.toLocaleString();
    document.getElementById('forecastDescription').textContent = `Total CVEs: Published + Forecasted (${bestModelName} - Best Model)`;

    if (forecastData.model_rankings?.length > 0) {
        const bestModel = forecastData.model_rankings[0];
        document.getElementById('bestModel').textContent = bestModel.model_name;
        document.getElementById('bestAccuracy').textContent = `${(bestModel.mape || 0).toFixed(2)}%`;
    }

    document.getElementById('totalCVEs').textContent = (forecastData.summary?.total_historical_cves || 0).toLocaleString();
}

/**
 * Populates the model selector dropdown.
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

/**
 * Populates the model performance rankings table.
 */
function populateModelRankings() {
    const tableBody = document.getElementById('modelRankingsTable');
    if (!tableBody) return;
    tableBody.innerHTML = '';

    forecastData.model_rankings?.forEach((model, index) => {
        const row = document.createElement('tr');
        row.className = `border-b border-gray-200 text-sm ${index % 2 === 0 ? 'bg-white' : 'bg-gray-50'} hover:bg-blue-50`;
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
            <td class="py-2 px-4 text-center font-mono">${index + 1}</td>
            <td class="py-2 px-4 font-medium text-gray-800">${model.model_name}</td>
            <td class="py-2 px-4 text-right font-mono">${(model.mape || 0).toFixed(2)}%</td>
            <td class="py-2 px-4 text-right font-mono">${(model.mase || 0).toFixed(2)}</td>
            <td class="py-2 px-4 text-right font-mono">${(model.rmsse || 0).toFixed(2)}</td>
            <td class="py-2 px-4 text-right font-mono">${(model.mae || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td class="py-2 px-4 text-center">
                <span class="inline-flex px-2 py-1 text-xs font-semibold leading-5 rounded-full ${badgeClass}">${performanceBadge}</span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

/**
 * Populates the 'Forecast vs Published' table with collapsible year sections.
 */
function populateForecastVsPublishedTable() {
    const selector = document.getElementById('validationModelSelector');
    const selectedModel = selector.value;
    const tableBody = document.getElementById('forecastVsPublishedTableBody');
    const maeCard = document.getElementById('validationMae');
    const mapeCard = document.getElementById('validationMape');

    if (!forecastData.forecast_vs_published || !forecastData.forecast_vs_published[selectedModel]) {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center py-4">No data available for this model.</td></tr>';
        maeCard.textContent = '-';
        mapeCard.textContent = '-';
        return;
    }

    const modelData = forecastData.forecast_vs_published[selectedModel];
    const tableData = modelData.table_data;

    const summaryStats = modelData.summary_stats;

    maeCard.textContent = (summaryStats.mean_absolute_error || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const bestModel = forecastData.model_rankings[0];
    mapeCard.textContent = `${(bestModel.mape || 0).toFixed(2)}%`;

    const groupedData = tableData.reduce((acc, row) => {
        const year = row.MONTH.split('-')[0];
        if (!acc[year]) {
            acc[year] = [];
        }
        acc[year].push(row);
        return acc;
    }, {});

    tableBody.innerHTML = '';
    const currentDisplayYear = new Date().getFullYear().toString();
    const sortedYears = Object.keys(groupedData).sort((a, b) => b - a);

    sortedYears.forEach(year => {
        const months = groupedData[year].sort((a, b) => new Date(b.MONTH) - new Date(a.MONTH));
        const isCollapsed = year !== currentDisplayYear;

        const headerRow = document.createElement('tr');
        headerRow.className = 'year-header';
        if (isCollapsed) headerRow.classList.add('collapsed');

        headerRow.innerHTML = `
            <td colspan="6" class="py-2 px-4 font-bold text-gray-700">
                <span class="toggle-icon">${isCollapsed ? '▶' : '▼'}</span>${year}
            </td>
        `;
        tableBody.appendChild(headerRow);

        months.forEach(row => {
            const dataRow = document.createElement('tr');
            dataRow.className = `month-row year-${year}`;
            if (isCollapsed) {
                dataRow.classList.add('hidden-row');
            }
            const error = row.ERROR;
            const percentError = row.PERCENT_ERROR;

            let errorColorClass = 'text-gray-800';
            if (error > 0) errorColorClass = 'text-red-600'; // Over-forecast
            if (error < 0) errorColorClass = 'text-green-600'; // Under-forecast

            let badgeClass = 'bg-red-100 text-red-800';
            let performanceBadge = 'Poor';
            const absPercentError = Math.abs(percentError);

            if (absPercentError < 10) {
                badgeClass = 'bg-green-100 text-green-800';
                performanceBadge = 'Excellent';
            } else if (absPercentError < 15) {
                badgeClass = 'bg-blue-100 text-blue-800';
                performanceBadge = 'Good';
            } else if (absPercentError < 25) {
                badgeClass = 'bg-yellow-100 text-yellow-800';
                performanceBadge = 'Fair';
            }

            dataRow.innerHTML = `
                <td class="py-2 px-4 font-mono">${row.MONTH}</td>
                <td class="text-right py-2 px-4 font-mono">${row.PUBLISHED.toLocaleString()}</td>
                <td class="text-right py-2 px-4 font-mono">${row.FORECAST.toLocaleString()}</td>
                <td class="text-right py-2 px-4 font-mono ${errorColorClass}">${error.toLocaleString()}</td>
                <td class="text-right py-2 px-4 font-mono ${errorColorClass}">${percentError.toFixed(2)}%</td>
                <td class="py-2 px-4 text-center font-mono"><span class="inline-flex px-2 py-1 text-xs font-semibold leading-5 rounded-full ${badgeClass}">${performanceBadge}</span></td>
            `;
            tableBody.appendChild(dataRow);
        });

        headerRow.addEventListener('click', () => {
            const icon = headerRow.querySelector('.toggle-icon');
            const isNowCollapsed = headerRow.classList.toggle('collapsed');
            icon.textContent = isNowCollapsed ? '▶' : '▼';

            document.querySelectorAll(`.month-row.year-${year}`).forEach(row => {
                row.classList.toggle('hidden-row', isNowCollapsed);
            });
        });
    });
}

/**
 * Updates the data period information cards.
 */
function updateDataPeriodInfo() {
    const summary = forecastData.summary;
    if (!summary) return;

    const formatDate = (dateStr) => new Date(dateStr).toLocaleDateString('en-US', { year: 'numeric', month: 'long', timeZone: 'UTC' });

    document.getElementById('historicalPeriod').textContent = `${formatDate(summary.data_period.start)} - ${formatDate(summary.data_period.end)}`;
    document.getElementById('forecastPeriod').textContent = `${formatDate(summary.forecast_period.start)} - ${formatDate(summary.forecast_period.end)}`;
}

/**
 * Creates or updates the main forecast chart.
 */
function createOrUpdateChart() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    const chartData = prepareChartData();
    const chartOptions = getChartOptions();

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: chartOptions,
    });
}

/**
 * Prepares the datasets for the chart, including actuals and all forecasts.
 */
function prepareChartData() {
    const { actuals_cumulative, cumulative_timelines } = forecastData;
    const datasets = [];

    const actualsData = actuals_cumulative.map(d => ({ x: new Date(d.date), y: d.cumulative_total }));

    datasets.push({
        label: 'Actual CVEs',
        data: actualsData,
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 3,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        tension: 0.1,
        fill: false,
    });

    const { model_rankings } = forecastData;
    const bestColor = [22, 163, 74];
    const worstColor = [200, 200, 200];

    const interpolateColor = (color1, color2, factor) => {
        const result = color1.slice();
        for (let i = 0; i < 3; i++) {
            result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
        }
        return `rgb(${result.join(', ')})`;
    };

    const topFiveModels = model_rankings.filter(m => cumulative_timelines[m.model_name + '_cumulative']).slice(0, 5);

    topFiveModels.forEach((model, index) => {
        const modelKey = `${model.model_name}_cumulative`;
        const modelData = cumulative_timelines[modelKey].map(d => ({ x: new Date(d.date), y: d.cumulative_total }));
        const factor = topFiveModels.length > 1 ? index / (topFiveModels.length - 1) : 0;
        const color = interpolateColor(bestColor, worstColor, factor);

        datasets.push({
            label: `${model.model_name} (Forecast)`,
            data: modelData,
            borderColor: color,
            borderWidth: 2,
            pointBackgroundColor: color,
            borderDash: [5, 5],
            tension: 0.1,
            fill: false,
            hidden: index !== 0,
        });
    });

    if (cumulative_timelines.all_models_cumulative) {
        const avgData = cumulative_timelines.all_models_cumulative.map(d => ({ x: new Date(d.date), y: d.cumulative_total }));
        datasets.push({
            label: 'Model Average (Forecast)',
            data: avgData,
            borderColor: 'rgb(239, 68, 68)',
            borderWidth: 2,
            pointBackgroundColor: 'rgb(239, 68, 68)',
            borderDash: [5, 5],
            tension: 0.1,
            fill: false,
            hidden: false,
        });
    }

    console.log(`Chart prepared with ${datasets.length} datasets.`);
    return { datasets };
}

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