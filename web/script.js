/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions.
 */

// Global state variables
let forecastData = null;
let modelInfoData = null;
let chartInstance = null;

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', loadForecastData);

/**
 * Loads forecast data from the data.json file and initializes the dashboard.
 */
async function loadForecastData() {
    console.log('🔄 Loading application data...');
    try {
        const [forecastResponse, modelInfoResponse] = await Promise.all([
            fetch('data.json?v=' + new Date().getTime()), // Cache-busting for forecast data
            fetch('model_info.json?v=' + new Date().getTime()) // Cache-busting for model info
        ]);

        if (!forecastResponse.ok) {
            throw new Error(`HTTP error! Status: ${forecastResponse.status} on data.json`);
        }
        if (!modelInfoResponse.ok) {
            console.warn(`Could not load model_info.json. Status: ${modelInfoResponse.status}. Links will not be available.`);
            modelInfoData = {}; // Set to empty object to prevent errors
        } else {
            try {
                modelInfoData = await modelInfoResponse.json();
            } catch (e) {
                console.error('Error parsing model_info.json:', e);
                modelInfoData = {};
            }
        }

        try {
            forecastData = await forecastResponse.json();
        } catch (e) {
            console.error('Error parsing data.json:', e);
            throw new Error('Failed to parse data.json');
        }
        console.log('✅ Application data loaded successfully.');

        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');

        initializeDashboard();
        console.log('✅ Dashboard initialized successfully!');

    } catch (error) {
        console.error('❌ Error in loadForecastData:', error);
        if (error instanceof Error) {
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
        }
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

    const lastYearTotal = forecastData.summary?.cumulative_cves_2024;
    if (bestModelTotal && lastYearTotal) {
        const yoyGrowth = ((bestModelTotal - lastYearTotal) / lastYearTotal) * 100;
        document.getElementById('yoyGrowth').textContent = `${yoyGrowth.toFixed(2)}%`;
    } else {
        document.getElementById('yoyGrowth').textContent = '-';
    }
}

/**
 * Populates the model selector dropdown.
 */
function populateModelSelector() {
    const selector = document.getElementById('validationModelSelector');
    if (!selector) return;
    selector.innerHTML = '';
    
    // Get the top 5 models from the rankings
    const topModels = forecastData.model_rankings?.slice(0, 5) || [];

    topModels.forEach(model => {
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

        const modelUrl = modelInfoData ? modelInfoData[model.model_name] : null;
        const modelNameHtml = modelUrl 
            ? `<a href="${modelUrl}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline">${model.model_name}</a>` 
            : model.model_name;

        row.innerHTML = `
            <td class="text-center font-mono">${index + 1}</td>
            <td class="font-medium text-gray-800">${modelNameHtml}</td>
            <td class="text-right font-mono">${(model.mape || 0).toFixed(2)}%</td>
            <td class="text-right font-mono">${(model.mae || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td class="text-center">
                <span class="inline-flex px-2 py-1 text-xs font-semibold leading-5 rounded-full ${badgeClass}">${performanceBadge}</span>
            </td>
            <td class="text-center">
                <button class="expand-btn" onclick="toggleConfig(${index})" id="expandBtn${index}">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                    </svg>
                </button>
            </td>
        `;
        
        tableBody.appendChild(row);

        // Create config row (initially hidden)
        const configRow = document.createElement('tr');
        configRow.id = `configRow${index}`;
        configRow.className = 'config-row hidden-row';
        configRow.innerHTML = `
            <td colspan="8" class="px-6 py-4">
                <div class="mb-2 flex justify-between items-center">
                    <h4 class="font-medium text-gray-800">Configuration for ${model.model_name}</h4>
                    <button class="copy-btn" onclick="copyConfig(${index})" id="copyBtn${index}">
                        Copy JSON
                    </button>
                </div>
                <div class="config-display" id="configDisplay${index}">
                    <!-- Content will be set via JavaScript -->
                </div>
            </td>
        `;
        tableBody.appendChild(configRow);
        
        // Set the configuration content after the element is in the DOM
        const configDisplay = document.getElementById(`configDisplay${index}`);
        if (configDisplay) {
            configDisplay.innerHTML = formatModelConfig(model);
        }
    });
}

/**
 * Populates the 'Forecast vs Published' table with collapsible year sections.
 */
function populateForecastVsPublishedTable() {
    const selector = document.getElementById('validationModelSelector');
    const selectedModel = selector.value;
    // Use correct IDs for summary cards and table body
    const tableBody = document.getElementById('validationTable');
    const maeCard = document.getElementById('avgErrorCard');
    const mapeCard = document.getElementById('avgPercentErrorCard');

    // Add null checks to avoid JS errors if elements are missing
    if (!tableBody || !maeCard || !mapeCard) {
        console.error('❌ Missing validation table or summary card elements in HTML.');
        return;
    }

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
        // Add Tailwind bg-gray-100 for year header shading
        headerRow.innerHTML = `
            <td colspan="6" class="font-bold text-gray-700 cursor-pointer">
                <span class="toggle-icon">${isCollapsed ? '▶' : '▼'}</span> ${year}
            </td>
        `;
        if (isCollapsed) headerRow.classList.add('collapsed');
        tableBody.appendChild(headerRow);

        months.forEach(row => {
            const dataRow = document.createElement('tr');
            dataRow.className = `month-row year-${year}`;
            if (isCollapsed) {
                dataRow.classList.add('hidden-row');
            }
            const now = new Date();
            const currentMonthStr = `${now.getFullYear()}-${(now.getMonth() + 1).toString().padStart(2, '0')}`;

            const formatMonth = (monthStr) => {
                const [year, month] = monthStr.split('-');
                const date = new Date(year, parseInt(month, 10) - 1);
                return date.toLocaleString('en-US', { month: 'long' });
            };

            if (row.MONTH === currentMonthStr) {
                dataRow.innerHTML = `
                    <td class="font-mono">${formatMonth(row.MONTH)}</td>
                    <td class="text-right font-mono">${row.PUBLISHED.toLocaleString()}</td>
                    <td class="text-right font-mono">${row.FORECAST.toLocaleString()}</td>
                    <td class="text-right font-mono text-gray-400" colspan="2"></td>
                    <td class="text-center font-mono"><span class="inline-flex px-2 py-1 text-xs font-semibold leading-5 rounded-full bg-gray-100 text-gray-800">In Progress</span></td>
                `;
            } else {
                const error = row.ERROR;
                const percentError = row.PERCENT_ERROR;

                const formatNumberWithSign = (num) => {
                    const sign = num > 0 ? '+' : '';
                    return `${sign}${num.toLocaleString()}`;
                };

                const formatPercentWithSign = (num) => {
                    const sign = num > 0 ? '+' : '';
                    return `${sign}${num.toFixed(2)}%`;
                };

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
                    <td class="font-mono">${formatMonth(row.MONTH)}</td>
                    <td class="text-right font-mono">${row.PUBLISHED.toLocaleString()}</td>
                    <td class="text-right font-mono">${row.FORECAST.toLocaleString()}</td>
                    <td class="text-right font-mono">${formatNumberWithSign(error)}</td>
                    <td class="text-right font-mono">${formatPercentWithSign(percentError)}</td>
                    <td class="text-center font-mono"><span class="inline-flex px-2 py-1 text-xs font-semibold leading-5 rounded-full ${badgeClass}">${performanceBadge}</span></td>
                `;
            }
            tableBody.appendChild(dataRow);
        });

        headerRow.addEventListener('click', () => {
            const icon = headerRow.querySelector('.toggle-icon');
            const isNowCollapsed = headerRow.classList.toggle('collapsed');
            icon.textContent = isNowCollapsed ? '▶' : '▼';
            // Toggle visibility of all month rows for this year
            const yearRows = tableBody.querySelectorAll(`.year-${year}`);
            yearRows.forEach(row => {
                if (isNowCollapsed) {
                    row.classList.add('hidden-row');
                } else {
                    row.classList.remove('hidden-row');
                }
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

                        // Check if it's the last point of the 'Actual CVEs' dataset
                        const isLastActualPoint = 
                            tooltipItems[0].datasetIndex === 0 && 
                            tooltipItems[0].dataIndex === forecastData.actuals_cumulative.length - 1;

                        if (isLastActualPoint) {
                            return pointDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
                        } else {
                            return pointDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' });
                        }
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
                max: '2026-01-05',
            },
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Number of CVEs' },
                ticks: { callback: (value) => value.toLocaleString() },
            },
        },
    };
}

/**
 * Toggles the visibility of a model configuration row.
 * @param {number} index - The index of the model in the rankings array
 */
function toggleConfig(index) {
    const configRow = document.getElementById(`configRow${index}`);
    const expandBtn = document.getElementById(`expandBtn${index}`);
    
    if (configRow.classList.contains('hidden-row')) {
        configRow.classList.remove('hidden-row');
        expandBtn.classList.add('expanded');
    } else {
        configRow.classList.add('hidden-row');
        expandBtn.classList.remove('expanded');
    }
}

/**
 * Copies the configuration JSON to clipboard.
 * @param {number} index - The index of the model in the rankings array
 */
async function copyConfig(index) {
    const model = forecastData.model_rankings[index];
    const config = formatModelConfigForDarts(model);
    const copyBtn = document.getElementById(`copyBtn${index}`);
    
    try {
        await navigator.clipboard.writeText(config);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.classList.remove('copied');
        }, 2000);
    } catch (err) {
        console.error('Failed to copy config:', err);
        // Fallback for browsers that don't support clipboard API
        const textArea = document.createElement('textarea');
        textArea.value = config;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => {
            copyBtn.textContent = 'Copy JSON';
            copyBtn.classList.remove('copied');
        }, 2000);
    }
}

/**
 * Formats model configuration for display.
 * @param {Object} model - The model object from rankings
 * @returns {string} Formatted configuration string
 */
function formatModelConfig(model) {
    // Priority: Show hyperparameters first if available, then performance metrics
    if (model.hyperparameters && Object.keys(model.hyperparameters).length > 0) {
        // Clean up hyperparameters by removing null values and formatting
        const cleanedHyperparameters = {};
        for (const [key, value] of Object.entries(model.hyperparameters)) {
            if (value !== null && value !== undefined) {
                cleanedHyperparameters[key] = value;
            }
        }
        
        const config = {
            model_name: model.model_name,
            hyperparameters: cleanedHyperparameters,
            split_ratio: parseFloat((model.split_ratio || 0.8).toFixed(3))
        };
        
        // Add tuning metadata if available
        if (model.tuned_at) {
            config.tuned_at = model.tuned_at;
        }
        
        // Add performance metrics as secondary info
        config.performance_metrics = {
            mape: parseFloat((model.mape || 0).toFixed(4)),
            mae: parseFloat((model.mae || 0).toFixed(4))
        };
        
        // Add training_time if available
        if (model.training_time !== undefined && model.training_time !== null) {
            config.performance_metrics.training_time = parseFloat(model.training_time.toFixed(6));
        }
        
        return syntaxHighlightJSON(JSON.stringify(config, null, 2).trim());
    } else {
        // Fallback for legacy data without hyperparameters
        const config = {
            model_name: model.model_name,
            split_ratio: parseFloat((model.split_ratio || 0.8).toFixed(3)),
            performance_metrics: {
                mape: parseFloat((model.mape || 0).toFixed(4)),
                mae: parseFloat((model.mae || 0).toFixed(4))
            },
            note: "Run comprehensive tuner to generate optimal hyperparameters"
        };
        
        // Add training_time if available
        if (model.training_time !== undefined && model.training_time !== null) {
            config.performance_metrics.training_time = parseFloat(model.training_time.toFixed(6));
        }
        
        return syntaxHighlightJSON(JSON.stringify(config, null, 2).trim());
    }
}

/**
 * Adds syntax highlighting to JSON string.
 * @param {string} json - The JSON string to highlight
 * @returns {string} HTML with syntax highlighting
 */
function syntaxHighlightJSON(json) {
    // Escape HTML first
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    })
    .replace(/([{}[\],])/g, '<span class="json-bracket">$1</span>');
}

/**
 * Formats model configuration for Darts usage.
 * @param {Object} model - The model object from rankings
 * @returns {string} Darts-compatible configuration string
 */
function formatModelConfigForDarts(model) {
    if (model.hyperparameters && Object.keys(model.hyperparameters).length > 0) {
        // Clean up hyperparameters by removing null values
        const cleanedHyperparameters = {};
        for (const [key, value] of Object.entries(model.hyperparameters)) {
            if (value !== null && value !== undefined) {
                cleanedHyperparameters[key] = value;
            }
        }
        
        // Full configuration with hyperparameters
        const dartsConfig = {
            model: model.model_name,
            hyperparameters: cleanedHyperparameters,
            split_ratio: parseFloat((model.split_ratio || 0.8).toFixed(3)),
            expected_performance: {
                mape: parseFloat((model.mape || 0).toFixed(4)),
                mae: parseFloat((model.mae || 0).toFixed(4))
            }
        };
        
        // Add training_time if available
        if (model.training_time !== undefined && model.training_time !== null) {
            dartsConfig.expected_performance.training_time = parseFloat(model.training_time.toFixed(6));
        }
        
        return JSON.stringify(dartsConfig, null, 2);
    } else {
        // Fallback for legacy data without hyperparameters
        const dartsConfig = {
            model: model.model_name,
            split_ratio: parseFloat((model.split_ratio || 0.8).toFixed(3)),
            expected_performance: {
                mape: parseFloat((model.mape || 0).toFixed(4)),
                mae: parseFloat((model.mae || 0).toFixed(4))
            },
            note: "No hyperparameters available - run comprehensive tuner for optimal parameters"
        };
        
        // Add training_time if available
        if (model.training_time !== undefined && model.training_time !== null) {
            dartsConfig.expected_performance.training_time = parseFloat(model.training_time.toFixed(6));
        }
        
        return JSON.stringify(dartsConfig, null, 2);
    }
}