/**
 * CVE Forecast Dashboard JavaScript
 * Handles data loading, visualization, and user interactions.
 */

// Theme toggle with localStorage persistence
(function initTheme() {
    const toggle = document.getElementById('themeToggle');
    const sunIcon = document.getElementById('themeSun');
    const moonIcon = document.getElementById('themeMoon');

    function applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (sunIcon) sunIcon.classList.remove('hidden');
            if (moonIcon) moonIcon.classList.add('hidden');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            if (sunIcon) sunIcon.classList.add('hidden');
            if (moonIcon) moonIcon.classList.remove('hidden');
        }
    }

    // Check saved preference, then system preference
    const saved = localStorage.getItem('theme');
    if (saved) {
        applyTheme(saved);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        applyTheme('dark');
    }

    if (toggle) {
        toggle.addEventListener('click', function() {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const newTheme = isDark ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
})();

// Constants
const TOP_MODELS_COUNT = 5;
const CHART_ANIMATION_DURATION = 750;
const CACHE_BUSTER = new Date().getTime();

// Global state variables
let forecastData = null;
let modelInfoData = null;
let chartInstance = null;
let selectedYear = new Date().getFullYear(); // Default to current year

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', loadForecastData);

/**
 * Loads forecast data from the data.json file and initializes the dashboard.
 */
async function loadForecastData() {
    console.log('🔄 Loading application data...');
    try {
        const [forecastResponse, modelInfoResponse] = await Promise.all([
            fetch('data.json?v=' + CACHE_BUSTER), // Cache-busting for forecast data
            fetch('model_info.json?v=' + CACHE_BUSTER) // Cache-busting for model info
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
    renderYearButtons();
    updateSummaryCards();
    renderMethodology();
    populateModelSelector();
    populateModelRankings();
    populateForecastVsPublishedTable(); // Initially populate with the best model
    updateDataPeriodInfo();
    updateChartDescription();
    createOrUpdateChart();

    const validationModelSelector = document.getElementById('validationModelSelector');
    if (validationModelSelector) {
        validationModelSelector.addEventListener('change', populateForecastVsPublishedTable);
    }
}

/**
 * Switches the chart and summary cards to a given forecast year.
 * @param {number} year
 */
function switchYear(year) {
    selectedYear = year;
    renderYearButtons();
    updateChartDescription();
    updateSummaryCards();
    createOrUpdateChart();
}

/**
 * Renders the year toggle. v0.11 looked for buttons that did not exist in the
 * markup, so the next year could never be viewed at all.
 */
function renderYearButtons() {
    const years = Object.keys(forecastData?.yearly_forecast_totals || {})
        .map(Number)
        .filter(y => (forecastData.yearly_forecast_totals[y]?.Ensemble?.months_forecast ?? 0) > 0)
        .sort((a, b) => a - b);
    if (years.length === 0) return;

    const active = 'bg-blue-600 text-white hover:bg-blue-700';
    const idle = 'bg-gray-200 text-gray-700 hover:bg-gray-300';
    [['yearBtnCurrent', years[0]], ['yearBtnNext', years[1]]].forEach(([id, year]) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        if (year === undefined) { btn.classList.add('hidden'); return; }
        btn.classList.remove('hidden');
        btn.textContent = year;
        btn.className = `year-btn px-4 py-2 rounded-lg font-semibold transition-colors ${year === selectedYear ? active : idle}`;
        btn.setAttribute('aria-pressed', String(year === selectedYear));
        btn.onclick = () => switchYear(year);
    });
}

function updateChartDescription() {
    const projection = forecastData?.yearly_forecast_totals?.[selectedYear]?.Ensemble;
    const el = document.getElementById('chartDescription');
    if (!el) return;

    if (projection && projection.months_actual > 0) {
        // Say plainly where the forecast starts: it continues from the count as of
        // now, including the part of the current month already published.
        el.textContent =
            `Cumulative CVE publications for ${selectedYear}. The forecast continues from the count as of today ` +
            `(${projection.actual_ytd.toLocaleString()} published); the shaded band is the 80% range.`;
    } else {
        el.textContent = `Forecast cumulative CVE publications for ${selectedYear}, with the 80% range shaded.`;
    }
}

/**
 * Updates the summary cards.
 *
 * The headline is no longer a pure model output: published months are counted as
 * facts and only the remainder is forecast, so the number tightens as the year
 * fills in. The 80% band is shown alongside it rather than a bare point estimate.
 */
function updateSummaryCards() {
    if (!forecastData) return;

    document.getElementById('lastUpdated').textContent =
        `Last Updated: ${new Date(forecastData.generated_at).toLocaleString()}`;

    const rankings = forecastData.model_rankings || [];
    const best = rankings.find(m => !m.is_baseline);
    const projection = forecastData.yearly_forecast_totals?.[selectedYear]?.Ensemble;

    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    if (projection) {
        setText('currentYearForecast', projection.total.toLocaleString());
        setText('forecastDescription',
            `${projection.actual_ytd.toLocaleString()} published (${projection.months_actual} mo) ` +
            `+ ${projection.forecast_remainder.toLocaleString()} forecast (${projection.months_forecast} mo)`);
    } else {
        setText('currentYearForecast', '-');
        setText('forecastDescription', 'No projection for this year');
    }

    if (best) {
        setText('bestModel', best.model_name);
        setText('bestAccuracy', best.mase != null ? best.mase.toFixed(2) : '-');
        const threshold = best.naive_threshold;
        setText('accuracyDetail', threshold
            ? `MASE over ${best.n_origins} origins · naive baseline ${threshold.toFixed(2)}`
            : `MASE over ${best.n_origins} origins`);
    }

    setText('totalCVEs', (forecastData.summary?.total_historical_cves || 0).toLocaleString());

    // Year-over-year against the previous year's settled total.
    const prev = forecastData.yearly_forecast_totals?.[selectedYear - 1]?.Ensemble;
    const prevTotal = prev?.total ?? forecastData.summary?.previous_year_total ?? 0;
    if (projection && prevTotal) {
        const growth = ((projection.total - prevTotal) / prevTotal) * 100;
        setText('yoyGrowth', `${growth >= 0 ? '+' : ''}${growth.toFixed(1)}%`);
        const band = projection.lower_80 != null
            ? ` · 80% range ${projection.lower_80.toLocaleString()}–${projection.upper_80.toLocaleString()}`
            : '';
        setText('yoyGrowthDetail',
            `${projection.total.toLocaleString()} vs ${prevTotal.toLocaleString()} (${selectedYear} vs ${selectedYear - 1})${band}`);
    } else {
        setText('yoyGrowth', '-');
        setText('yoyGrowthDetail', 'Data unavailable');
    }
}

/**
 * Renders the methodology and interval-calibration panel.
 */
function renderMethodology() {
    const panel = document.getElementById('methodologyPanel');
    const meta = forecastData?.methodology;
    if (!panel || !meta) return;

    const rankings = forecastData.model_rankings || [];
    const scored = rankings.filter(m => !m.is_baseline && m.mase != null);
    const winners = scored.filter(m => m.beats_naive);

    const coverage = meta.interval_coverage || {};
    const coverageHtml = Object.keys(coverage).length
        ? Object.entries(coverage).map(([level, stats]) =>
            `<div class="flex justify-between text-sm py-1">
                 <span class="text-gray-600">${level}% interval</span>
                 <span class="font-mono font-semibold ${stats.calibrated ? 'text-green-700' : 'text-red-700'}">
                     ${(stats.empirical * 100).toFixed(1)}% covered
                 </span>
             </div>`).join('')
        : '<p class="text-sm text-gray-500">Not yet available</p>';

    const card = (title, body, note) => `
        <div class="border border-gray-200 rounded-lg p-4">
            <h3 class="font-semibold text-gray-800 mb-2">${title}</h3>
            ${body}
            ${note ? `<p class="text-xs text-gray-500 mt-2">${note}</p>` : ''}
        </div>`;

    panel.innerHTML = [
        card('Models vs. naive baseline',
             `<p class="text-2xl font-bold text-gray-900">${winners.length} of ${scored.length}</p>`,
             `beat the naive benchmark (MASE ${meta.naive_threshold?.toFixed(2) ?? 'n/a'}). The rest are shown but not used.`),
        card('Interval calibration', coverageHtml,
             'How often the published band actually contained the outcome, in backtest.'),
        card('Ensemble',
             `<p class="text-sm text-gray-800">${(meta.ensemble_members || []).join(', ') || 'none'}</p>`,
             'Trimmed mean over the models that cleared the baseline.'),
    ].join('');

    const s = meta.settings || {};
    const settingsEl = document.getElementById('methodologySettings');
    if (settingsEl) {
        settingsEl.textContent =
            `Pipeline: ${s.log_space ? 'log-space' : 'levels'}, ` +
            `${s.business_day_normalise ? 'business-day normalised' : 'raw monthly'}, ` +
            `trend damping φ=${s.damping_phi}, ` +
            `training window ${s.training_window_months ? s.training_window_months + ' months' : 'full history'}. ` +
            `Ranking metric: ${meta.ranking_metric}.`;
    }
}

function populateModelSelector() {
    const selector = document.getElementById('validationModelSelector');
    if (!selector) return;
    selector.innerHTML = '';
    
    // Get the top models from the rankings
    const topModels = (forecastData.model_rankings || []).filter(m => !m.is_baseline).slice(0, TOP_MODELS_COUNT);

    topModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.model_name;
        option.textContent = model.model_name;
        selector.appendChild(option);
    });
}

/**
 * Populates the model rankings table.
 *
 * v0.11 declared seven columns and emitted six cells, so every value after MAPE
 * rendered one column to the left of its header.
 */
function populateModelRankings() {
    const tableBody = document.getElementById('modelRankingsTable');
    if (!tableBody) return;
    tableBody.innerHTML = '';

    const basis = document.getElementById('rankingBasis');
    const anyModel = forecastData.model_rankings?.[0];
    if (basis && anyModel) {
        basis.textContent = `${anyModel.n_origins} rolling origins · lower MASE is better`;
    }

    forecastData.model_rankings?.forEach((model, index) => {
        const row = document.createElement('tr');

        let badgeClass, verdict;
        if (model.is_baseline) {
            badgeClass = 'pill--neutral';
            verdict = 'Baseline';
        } else if (model.mase == null) {
            badgeClass = 'pill--bad';
            verdict = 'Failed';
        } else if (model.beats_naive) {
            badgeClass = 'pill--good';
            verdict = 'Beats naive';
        } else {
            badgeClass = 'pill--warn';
            verdict = 'Loses to naive';
        }

        const modelUrl = modelInfoData ? modelInfoData[model.model_name] : null;
        const modelNameHtml = modelUrl
            ? `<a href="${modelUrl}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline">${model.model_name}</a>`
            : model.model_name;
        const ensembleTag = model.in_ensemble
            ? ' <span class="text-xs text-green-700 font-semibold" title="Included in the published ensemble">&#9733;</span>'
            : '';

        const fmt = (v, digits, suffix = '') => (v == null ? '—' : v.toFixed(digits) + suffix);
        const bias = model.bias_pct == null ? '—' : `${model.bias_pct >= 0 ? '+' : ''}${model.bias_pct.toFixed(1)}%`;

        row.innerHTML = `
            <td class="text-center font-mono">${index + 1}</td>
            <td class="font-medium text-gray-800">${modelNameHtml}${ensembleTag}</td>
            <td class="text-right font-mono">${fmt(model.mase, 2)}${model.mase_std != null ? ` <span class="text-gray-400 text-xs">±${model.mase_std.toFixed(2)}</span>` : ''}</td>
            <td class="text-right font-mono">${fmt(model.mape, 1, '%')}</td>
            <td class="text-right font-mono">${bias}</td>
            <td class="text-center">
                <span class="pill ${badgeClass}">${verdict}</span>
            </td>
            <td class="text-center">
                <button class="expand-btn" onclick="toggleConfig(${index})" id="expandBtn${index}" aria-label="Show configuration for ${model.model_name}">
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
    mapeCard.textContent = `${(summaryStats.mean_absolute_percentage_error || 0).toFixed(2)}%`;

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
                    <td class="text-center"><span class="pill pill--neutral">In Progress</span></td>
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

                let badgeClass = 'pill--bad';
                let performanceBadge = 'Poor';
                const absPercentError = Math.abs(percentError);

                if (absPercentError < 10) {
                    badgeClass = 'pill--good';
                    performanceBadge = 'Excellent';
                } else if (absPercentError < 15) {
                    badgeClass = 'pill--info';
                    performanceBadge = 'Good';
                } else if (absPercentError < 25) {
                    badgeClass = 'pill--warn';
                    performanceBadge = 'Fair';
                }

                dataRow.innerHTML = `
                    <td class="font-mono">${formatMonth(row.MONTH)}</td>
                    <td class="text-right font-mono">${row.PUBLISHED.toLocaleString()}</td>
                    <td class="text-right font-mono">${row.FORECAST.toLocaleString()}</td>
                    <td class="text-right font-mono">${formatNumberWithSign(error)}</td>
                    <td class="text-right font-mono">${formatPercentWithSign(percentError)}</td>
                    <td class="text-center"><span class="pill ${badgeClass}">${performanceBadge}</span></td>
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
        chartInstance.data = chartData;
        chartInstance.options = chartOptions;
        chartInstance.update('none'); // 'none' = no animation on update
    } else {
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: chartOptions,
        });
    }
}

/**
 * Prepares the datasets for the chart, including actuals and all forecasts.
 * Filters data based on the selected year.
 */
function prepareChartData() {
    const { actuals_cumulative, cumulative_timelines } = forecastData;
    const datasets = [];

    // Filter actuals data for selected year (Jan 1 through current date or Dec 31)
    // Use UTC parsing to avoid timezone conversion issues
    const actualsData = actuals_cumulative
        .filter(d => {
            const dateStr = d.date.substring(0, 4); // Extract year as string "2025"
            return parseInt(dateStr) === selectedYear;
        })
        .map(d => ({ x: new Date(d.date), y: d.cumulative_total }));

    // Determine the last fully completed month (first-of-month entry) for the selected year
    const lastActualMonthEntry = actuals_cumulative
        .filter(d => d.date.startsWith(`${selectedYear}-`) && d.date.endsWith('-01T00:00:00Z'))
        .reduce((latest, current) => {
            if (!latest) return current;
            return current.date > latest.date ? current : latest;
        }, null);
    const lastActualMonthDateStr = lastActualMonthEntry ? lastActualMonthEntry.date : null;
    
    console.log(`📊 Actuals data for ${selectedYear}:`, actualsData.length, 'points');
    if (actualsData.length > 0) {
        console.log('  First:', actualsData[0]);
        console.log('  Last:', actualsData[actualsData.length - 1]);
    }

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

    const topFiveModels = model_rankings
        .filter(m => !m.is_baseline && cumulative_timelines[m.model_name + '_cumulative'])
        .slice(0, TOP_MODELS_COUNT);

    topFiveModels.forEach((model, index) => {
        const modelKey = `${model.model_name}_cumulative`;
        // Filter forecast data by selected year
        // Use UTC parsing to avoid timezone conversion issues
        const modelData = cumulative_timelines[modelKey]
            .filter(d => {
                const dateStr = d.date.substring(0, 4); // Extract year as string "2025"
                const isSelectedYear = parseInt(dateStr) === selectedYear;
                // Exclude year boundary reset markers (Dec 31 with value 0, which belong to next year's view)
                const isResetMarker = d.date.includes('-12-31T23:59:59Z') && d.cumulative_total === 0;
                const entryDateStr = d.date;
                const isAfterActuals = !lastActualMonthDateStr || entryDateStr >= lastActualMonthDateStr;
                return isSelectedYear && !isResetMarker && isAfterActuals;
            })
            .map(d => ({ x: new Date(d.date), y: d.cumulative_total }));
        
        if (index === 0) {
            console.log(`📈 Forecast data for ${model.model_name} (${selectedYear}):`, modelData.length, 'points');
            if (modelData.length > 0) {
                console.log('  First:', modelData[0]);
                console.log('  Last:', modelData[modelData.length - 1]);
            }
        }
        
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

    // Ensemble line plus its 80% prediction band. v0.11 published bare point
    // estimates 16 months out; the band is the honest version of that claim.
    const ensembleTimeline = cumulative_timelines.Ensemble_cumulative;
    if (ensembleTimeline) {
        // >= so the forecast path starts at the anchor point carrying the count as
        // of now, continuing the actuals line instead of floating a month later.
        const inYear = d => {
            const isReset = d.date.includes('-12-31T23:59:59Z') && d.cumulative_total === 0;
            const afterActuals = !lastActualMonthDateStr || d.date >= lastActualMonthDateStr;
            return parseInt(d.date.substring(0, 4)) === selectedYear && !isReset && afterActuals;
        };
        const points = ensembleTimeline.filter(inYear);
        const band = cumulativeBandFor(inYear);

        if (band) {
            datasets.push({
                label: '80% range',
                data: band.upper,
                borderColor: 'rgba(239, 68, 68, 0.2)',
                backgroundColor: 'rgba(239, 68, 68, 0.10)',
                borderWidth: 0,
                pointRadius: 0,
                fill: '+1',
                tension: 0.1,
            });
            datasets.push({
                label: '80% range (lower bound)',
                data: band.lower,
                borderColor: 'rgba(239, 68, 68, 0.2)',
                borderWidth: 0,
                pointRadius: 0,
                fill: false,
                tension: 0.1,
            });
        }

        datasets.push({
            label: 'Ensemble (Forecast)',
            data: points.map(d => ({ x: new Date(d.date), y: d.cumulative_total })),
            borderColor: 'rgb(239, 68, 68)',
            borderWidth: 2,
            pointBackgroundColor: 'rgb(239, 68, 68)',
            borderDash: [5, 5],
            tension: 0.1,
            fill: false,
        });
    }

    console.log(`Chart prepared with ${datasets.length} datasets.`);
    return { datasets };
}

/**
 * Reads the server-computed 80% cumulative band for the selected year.
 *
 * The maths lives in the pipeline, not here: deriving which month each cumulative
 * step belongs to in the browser is easy to get off by one, and a silently
 * mislabelled band is worse than no band at all.
 *
 * @param {(d: {date: string, cumulative_total: number}) => boolean} inYear Filter applied to the ensemble line
 * @returns {{lower: Array, upper: Array}|null} Plottable points, or null if unavailable
 */
function cumulativeBandFor(inYear) {
    const band = forecastData?.cumulative_band;
    if (!band?.lower?.length || !band?.upper?.length) return null;

    const toPoints = rows => rows.filter(inYear).map(d => ({ x: new Date(d.date), y: d.cumulative_total }));
    const lower = toPoints(band.lower);
    const upper = toPoints(band.upper);
    return lower.length && upper.length ? { lower, upper } : null;
}

/**
 * Returns the configuration options for the chart.
 */
function getChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                    position: 'top',
                    // The band's lower bound is a fill artefact, not a series to toggle.
                    labels: { usePointStyle: true, padding: 20, filter: item => !item.text.includes('(lower bound)') },
                },
            tooltip: {
                mode: 'nearest',
                intersect: true,
                callbacks: {
                    title: (tooltipItems) => {
                        if (!tooltipItems.length) return '';
                        const pointDate = new Date(tooltipItems[0].parsed.x);

                        // Check if this is a Dec 31 23:59:59 timestamp (End of Year marker)
                        const month = pointDate.getUTCMonth();
                        const day = pointDate.getUTCDate();
                        const hour = pointDate.getUTCHours();
                        const minute = pointDate.getUTCMinutes();
                        const second = pointDate.getUTCSeconds();
                        
                        if (month === 11 && day === 31 && hour === 23 && minute === 59 && second === 59) {
                            return `End of Year ${pointDate.getUTCFullYear()}`;
                        }

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
                min: new Date(selectedYear - 1, 11, 25), // Dec 25 of previous year for padding
                max: new Date(selectedYear, 11, 31, 23, 59, 59), // Dec 31 at end of day
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