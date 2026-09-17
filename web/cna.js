// CNA Forecast JavaScript - Consolidated

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

// Global variables
let cnaData = {};
let cnaNameMapping = {};
let sortedCnaIds = [];
let chartInstance = null;
let currentYear = new Date().getFullYear();
let currentCnaData = null;

// Table variables
let tableData = [];
let filteredData = [];
let currentPage = 1;
let pageSize = 10;
let sortColumn = 'forecastedCurrent';
let sortDirection = 'desc';

/**
 * Years this page talks about, derived from the data rather than hard-coded.
 * Populated by deriveYears() once the payload loads.
 */
let YEARS = { priorYear: 0, currentYear: 0, nextYear: 0 };

/**
 * Works out which year is "current" from the data itself.
 *
 * Anchored on the most recent *historical* month across all CNAs, not on the
 * earliest forecast month: a CNA that has stopped publishing gets a forecast
 * starting from its own last data point, and four of them currently reach back
 * to 2022. Taking a global minimum over those put the whole page in 2022.
 *
 * @param {Object} payload Parsed cna_data.json
 * @returns {{priorYear: number, currentYear: number, nextYear: number}}
 */
function deriveYears(payload) {
  let latest = '';
  Object.values(payload || {}).forEach(rec => {
    Object.keys(rec?.historical || {}).forEach(month => {
      if (month > latest) latest = month;
    });
  });

  const currentYear = latest ? Number(latest.slice(0, 4)) : new Date().getFullYear();
  return { priorYear: currentYear - 1, currentYear, nextYear: currentYear + 1 };
}

/** Writes the derived years into every label that names one. */
function applyYearLabels() {
  const { priorYear, currentYear, nextYear } = YEARS;
  const set = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };
  set('summaryPriorYearLabel', `${priorYear} published`);
  set('summaryCurrentYearLabel', `${currentYear} projected`);
  set('summaryGrowthDetail', `${priorYear} → ${currentYear} change`);
  set('thPriorYear', `${priorYear} Published`);
  set('thCurrentYear', `${currentYear} Projected`);
  set('thNextYear', `${nextYear} Forecast`);
  set('thGrowth', `${priorYear}→${currentYear} Growth`);
  set('yearCurrentBtn', String(currentYear));
  set('yearNextBtn', String(nextYear));
}

// Formatting
const numberFmt = new Intl.NumberFormat();

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Load CNA data
async function loadCnaData() {
  try {
    // Add cache-busting parameter to force fresh data load
    const response = await fetch(`cna_data.json?v=${CACHE_BUSTER}`);
    
    if (response.ok) {
      const text = await response.text();
      
      try {
        cnaData = JSON.parse(text);

        // Must run before any row is processed: every year label and every
        // year-scoped total reads from YEARS.
        YEARS = deriveYears(cnaData);
        currentYear = YEARS.currentYear;
        applyYearLabels();

        // Sort CNAs by total historical CVE count
        sortedCnaIds = sortCnasByTotal(cnaData);
        
        // Initialize table
        initializeTable();
        
        // Update dynamic table headers
        updateDynamicHeaders();
        
        // Auto-select top CNA and display chart
        autoSelectTopCna();
        
        // Update model statistics after data is loaded with comprehensive logging
        console.log('loadCnaData: Data loaded successfully, scheduling updateModelStatistics');
        setTimeout(() => {
          console.log('loadCnaData: Timeout fired, calling updateModelStatistics');
          updateModelStatistics();
        }, 150);
      } catch (parseError) {
        console.error('CNA Data Parse Error:', parseError);
      }
    }
  } catch (error) {
    console.error('CNA Data Load Error:', error);
  }
}

// Fetch CNA name mapping from CNAScoreCard data with offline fallback
async function loadCnaNameMapping() {
  try {
    const response = await fetch('https://raw.githubusercontent.com/RogoLabs/CNAScoreCard/main/web/data/cna_list.json');
    if (response.ok) {
      const data = await response.json();
      cnaNameMapping = {};
      data.forEach(cna => {
        if (cna.shortName) {
          cnaNameMapping[cna.shortName.toLowerCase()] = {
            displayName: cna.organizationName || cna.name || cna.shortName,
            shortName: cna.shortName
          };
        }
      });
    }
  } catch (error) {
    // Using fallback names
  }
}

function getCnaDisplayName(cnaId, fallbackName) {
  if (!fallbackName) {
    return cnaId;
  }
  
  // First try exact match with fallback name
  let mapping = cnaNameMapping[fallbackName.toLowerCase()];
  if (mapping && mapping.displayName) {
    return mapping.displayName;
  }
  
  // Try to find by searching through all mappings for a match
  for (const [key, value] of Object.entries(cnaNameMapping)) {
    if (key.includes(fallbackName.toLowerCase()) || fallbackName.toLowerCase().includes(key)) {
      if (value && value.displayName) {
        return value.displayName;
      }
    }
  }
  
  // Capitalize the fallback name for better display
  return fallbackName.charAt(0).toUpperCase() + fallbackName.slice(1);
}

function getCnaShortName(cnaId, fallbackName) {
  if (!fallbackName) return cnaId.substring(0, 8);
  
  // First try exact match with fallback name
  let mapping = cnaNameMapping[fallbackName.toLowerCase()];
  if (mapping) {
    return mapping.shortName;
  }
  
  // Try to find by searching through all mappings for a match
  for (const [key, value] of Object.entries(cnaNameMapping)) {
    if (key.includes(fallbackName.toLowerCase()) || fallbackName.toLowerCase().includes(key)) {
      return value.shortName;
    }
  }
  
  // Return capitalized fallback name
  return fallbackName.charAt(0).toUpperCase() + fallbackName.slice(1);
}

function sortCnasByTotal(data) {
  return Object.entries(data)
    .map(([id, obj]) => [id, Object.values(obj.historical || {}).reduce((a,b) => a + b, 0)])
    .sort((a,b) => b[1] - a[1])
    .map(([id]) => id);
}

function toXY(dict, isHistorical = false) {
  const entries = Object.entries(dict || {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([ym, v]) => ({ x: ym + '-01', y: v }));
  
  if (isHistorical && entries.length > 0) {
    entries.unshift({ x: entries[0].x.replace(/-\d{2}-/, '-01-'), y: 0 });
  }
  
  return entries;
}

function calculateCnaMetrics(rec) {
  try {
    if (!rec) {
      throw new Error('Record is null or undefined');
    }
    
    if (!rec.id) {
      throw new Error('Record missing id field');
    }
    
    const displayName = getCnaDisplayName(rec.id, rec.name);
    const shortName = getCnaShortName(rec.id, rec.name);
    
    // Years are derived from the data, never hard-coded. The page previously
    // pinned 2024/2025/2026 into the markup, so by September 2026 it labelled the
    // settled 2025 total as a forecast and showed the current year one column
    // over. YEARS is computed once in deriveYears().
    const { priorYear, currentYear, nextYear } = YEARS;

    let priorTotal = 0;
    let currentPublished = 0;

    if (rec.historical) {
      Object.entries(rec.historical).forEach(([month, count]) => {
        if (typeof count !== 'number') return;
        const year = Number(month.slice(0, 4));
        if (year === priorYear) priorTotal += count;
        else if (year === currentYear) currentPublished += count;
      });
    }

    // The forecast starts mid-year, so summing only its months gives a partial
    // year. The current year is published-to-date plus the forecast remainder;
    // only the next year is forecast end to end.
    let currentRemainder = 0;
    let nextForecast = 0;
    const selectedModel = rec.model_selection?.selected_model;
    const modelForecasts = selectedModel ? rec.forecasts?.[selectedModel] : null;

    if (modelForecasts) {
      Object.entries(modelForecasts).forEach(([month, value]) => {
        if (typeof value !== 'number') return;
        const year = Number(month.slice(0, 4));
        if (year === currentYear) currentRemainder += value;
        else if (year === nextYear) nextForecast += value;
      });
    }

    const forecastedCurrent = currentPublished + Math.round(currentRemainder);
    const forecastedNext = Math.round(nextForecast);
    const growthRate = priorTotal > 0 ? ((forecastedCurrent - priorTotal) / priorTotal) * 100 : 0;

    const result = {
      id: rec.id,
      name: displayName,
      shortName: shortName,
      priorTotal: priorTotal,
      currentPublished: currentPublished,
      currentRemainder: Math.round(currentRemainder),
      forecastedCurrent: forecastedCurrent,
      forecastedNext: forecastedNext,
      growthRate: growthRate,
      model: rec.model_selection?.selected_model || 'N/A',
      mase: rec.model_selection?.validation_mase ?? null,
      isFallback: rec.model_selection?.is_fallback === true
    };

    return result;
  } catch (error) {
    throw error;
  }
}

// =============================================================================
// TABLE FUNCTIONS
// =============================================================================

function initializeTable() {
  try {
    if (!cnaData || Object.keys(cnaData).length === 0) {
      return;
    }
    tableData = Object.values(cnaData).map((rec) => {
      try {
        return calculateCnaMetrics(rec);
      } catch (metricError) {
        throw metricError;
      }
    });
    
    filteredData = [...tableData];
    sortTable();
    renderTable();
    updatePagination();
    
    // Set up table sorting event listeners after table is initialized
    setupTableSorting();
    
    // Chart will be auto-selected by autoSelectTopCna() after table initialization
    
  } catch (error) {
    console.error('Table Initialization Error:', error);
  }
}

function sortTable() {
  filteredData.sort((a, b) => {
    let aVal = a[sortColumn];
    let bVal = b[sortColumn];
    
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }
    
    if (sortDirection === 'asc') {
      return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    } else {
      return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
    }
  });
}

function filterTable(searchTerm) {
  const term = searchTerm.toLowerCase();
  filteredData = tableData.filter(row => 
    row.name.toLowerCase().includes(term)
  );
  currentPage = 1;
  renderTable();
  updatePagination();
}

function renderTable() {
  const tbody = document.getElementById('cnaTableBody');
  const start = (currentPage - 1) * pageSize;
  const end = start + pageSize;
  const pageData = filteredData.slice(start, end);
  
  tbody.innerHTML = pageData.map(row => {
    const growthClass = row.growthRate > 0 ? 'text-green-600' : row.growthRate < 0 ? 'text-red-600' : 'text-gray-600';
    const growthSymbol = row.growthRate > 0 ? '+' : '';
    
    return `
      <tr class="hover:bg-gray-50 cursor-pointer" onclick="selectCnaFromTable('${row.id}')">
        <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">${row.name || 'Unknown CNA'}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.priorTotal)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.forecastedCurrent)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.forecastedNext)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm ${growthClass}">${growthSymbol}${row.growthRate.toFixed(1)}%</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
          <span class="pill pill--info pill--nodot">${row.model}</span>
        </td>
      </tr>
    `;
  }).join('');
}

function updatePagination() {
  const total = filteredData.length;
  const totalPages = Math.ceil(total / pageSize);
  const start = (currentPage - 1) * pageSize + 1;
  const end = Math.min(start + pageSize - 1, total);
  
  // Update pagination info using the correct element IDs from HTML
  const startElement = document.getElementById('tableShowingStart');
  const endElement = document.getElementById('tableShowingEnd');
  const totalElement = document.getElementById('tableTotal');
  
  if (startElement) startElement.textContent = start;
  if (endElement) endElement.textContent = end;
  if (totalElement) totalElement.textContent = total;
  
  // Update pagination buttons
  const prevBtn = document.getElementById('tablePrevBtn');
  const nextBtn = document.getElementById('tableNextBtn');
  const paginationNumbers = document.getElementById('tablePaginationNumbers');
  
  if (prevBtn) {
    prevBtn.disabled = currentPage <= 1;
    prevBtn.onclick = currentPage > 1 ? () => goToPage(currentPage - 1) : null;
  }
  
  if (nextBtn) {
    nextBtn.disabled = currentPage >= totalPages;
    nextBtn.onclick = currentPage < totalPages ? () => goToPage(currentPage + 1) : null;
  }
  
  if (paginationNumbers) {
    let numbersHTML = '';
    for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
      if (i === currentPage) {
        numbersHTML += `<button class="px-3 py-2 text-sm font-medium text-blue-600 bg-blue-50 border border-gray-300 rounded">${i}</button>`;
      } else {
        numbersHTML += `<button onclick="goToPage(${i})" class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded hover:bg-gray-50">${i}</button>`;
      }
    }
    paginationNumbers.innerHTML = numbersHTML;
  }
}

function goToPage(page) {
  currentPage = page;
  renderTable();
  updatePagination();
}

function selectCnaFromTable(cnaId) {
  const rec = cnaData[cnaId];
  if (rec) {
    currentCnaData = rec;
    updateSummary(rec);
    renderChart(rec);
  }
}

// Handle sort click events
function handleSort(event) {
  const header = event.currentTarget;
  const column = header.getAttribute('data-sort');
  
  // Toggle sort direction
  if (sortColumn === column) {
    sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
  } else {
    sortColumn = column;
    sortDirection = 'asc';
  }
  
  // Update sort indicators
  const headers = document.querySelectorAll('#cnaTable th[data-sort]');
  headers.forEach(h => {
    const indicator = h.querySelector('.sort-indicator');
    if (h === header) {
      indicator.textContent = sortDirection === 'asc' ? '↑' : '↓';
    } else {
      indicator.textContent = '↕';
    }
  });
  
  // Reset to page 1 when sorting
  currentPage = 1;
  
  // Sort and re-render table
  sortTable();
  renderTable();
  updatePagination();
}

// Set up table sorting functionality
function setupTableSorting() {
  const headers = document.querySelectorAll('#cnaTable th[data-sort]');
  headers.forEach(header => {
    // Remove existing listeners to prevent duplicates
    header.removeEventListener('click', handleSort);
    header.addEventListener('click', handleSort);
  });
  
  // Set up search functionality
  const searchInput = document.getElementById('tableSearch');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      filterTable(e.target.value);
    });
  }
  
  // Set up page size selector
  const pageSizeSelect = document.getElementById('tablePageSize');
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', (e) => {
      pageSize = parseInt(e.target.value);
      currentPage = 1;
      renderTable();
      updatePagination();
    });
  }
  
  // Initialize sort indicator for default column
  const defaultHeader = document.querySelector(`#cnaTable th[data-sort="${sortColumn}"]`);
  if (defaultHeader) {
    const indicator = defaultHeader.querySelector('.sort-indicator');
    if (indicator) {
      indicator.textContent = sortDirection === 'asc' ? '↑' : '↓';
    }
  }
}

// Update dynamic table headers based on current year
function updateDynamicHeaders() {
  const growthHeader = document.querySelector('#cnaTable th[data-sort="growthRate"]');
  if (growthHeader) {
    const previousYear = currentYear - 1;
    const headerText = `${previousYear}→${currentYear} Growth`;
    // Update the text content while preserving the sort indicator
    const indicator = growthHeader.querySelector('.sort-indicator');
    const indicatorText = indicator ? indicator.textContent : '↕';
    growthHeader.innerHTML = `${headerText} <span class="sort-indicator">${indicatorText}</span>`;
  }
}

// Auto-select top CNA and display its chart
function autoSelectTopCna() {
  console.log('autoSelectTopCna: Starting...');
  if (filteredData && filteredData.length > 0) {
    const topCna = filteredData[0];
    console.log('autoSelectTopCna: Top CNA:', topCna.name, 'ID:', topCna.id);
    const rec = cnaData[topCna.id];
    if (rec) {
      console.log('autoSelectTopCna: Found CNA record, updating summary and chart');
      currentCnaData = rec;
      updateSummary(rec);
      renderChart(rec);
    } else {
      console.log('autoSelectTopCna: CNA record not found for ID:', topCna.id);
    }
  } else {
    console.log('autoSelectTopCna: No filtered data available');
  }
}

// CHART FUNCTIONS
// =============================================================================

function getModelColor(modelName, alpha = 1) {
  const colors = {
    'Prophet': `rgba(168, 85, 247, ${alpha})`,
    'XGBoost': `rgba(34, 197, 94, ${alpha})`,
    'LightGBM': `rgba(251, 146, 60, ${alpha})`,
    'AutoARIMA': `rgba(59, 130, 246, ${alpha})`,
    'ExponentialSmoothing': `rgba(107, 114, 128, ${alpha})`,
    'CatBoost': `rgba(236, 72, 153, ${alpha})`,
    'RandomForest': `rgba(16, 185, 129, ${alpha})`,
    'LinearRegression': `rgba(99, 102, 241, ${alpha})`,
    'TBATS': `rgba(251, 146, 60, ${alpha})`,
    'Theta': `rgba(139, 69, 19, ${alpha})`,
    'FourTheta': `rgba(75, 85, 99, ${alpha})`,
    'KalmanFilter': `rgba(220, 38, 127, ${alpha})`,
    'Croston': `rgba(6, 182, 212, ${alpha})`,
    'TCN': `rgba(124, 58, 237, ${alpha})`,
    'NBEATS': `rgba(217, 70, 239, ${alpha})`,
    'NHiTS': `rgba(34, 197, 94, ${alpha})`,
    'TiDE': `rgba(245, 158, 11, ${alpha})`,
    'DLinear': `rgba(52, 211, 153, ${alpha})`
  };
  
  return colors[modelName] || `rgba(107, 114, 128, ${alpha})`;
}

function buildCumulativeDatasets(rec, year) {
  console.log('buildCumulativeDatasets: Starting for CNA:', rec.name || rec.id, 'Year:', year);
  const datasets = [];
  
  // Determine the last fully completed month (first-of-month entry) for the selected year
  // This is used to filter forecast data to only show points after the last historical month
  let lastActualMonthDateStr = null;
  if (rec.historical_cumulative && Array.isArray(rec.historical_cumulative)) {
    const lastActualMonthEntry = rec.historical_cumulative
      .filter(d => d.date.startsWith(`${year}-`) && d.date.endsWith('-01T00:00:00Z'))
      .reduce((latest, current) => {
        if (!latest) return current;
        return current.date > latest.date ? current : latest;
      }, null);
    lastActualMonthDateStr = lastActualMonthEntry ? lastActualMonthEntry.date : null;
  }
  
  // Historical data (actual CVEs published)
  // Include MTD point to match main CVE page behavior
  if (rec.historical_cumulative && Array.isArray(rec.historical_cumulative)) {
    const historicalData = rec.historical_cumulative
      .filter(item => {
        const dateStr = item.date.substring(0, 4); // Extract year as string
        return parseInt(dateStr) === year;
      })
      .map(item => ({
        x: new Date(item.date),
        y: item.cumulative_total
      }));
    
    console.log(`📊 Historical data for ${year}:`, historicalData.length, 'points');
    if (historicalData.length > 0) {
      console.log('  First:', historicalData[0]);
      console.log('  Last:', historicalData[historicalData.length - 1]);
      console.log('  Last complete month:', lastActualMonthDateStr);
    }
    
    if (historicalData.length > 0) {
      const historicalDataset = {
        label: 'Historical CVEs',
        data: historicalData,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointBackgroundColor: '#3b82f6',
        pointBorderColor: '#3b82f6',
        pointRadius: 3,
        tension: 0.1,
        fill: false
      };
      datasets.push(historicalDataset);
    }
  }
  
  // Forecast data - use cumulative_timelines structure (matching main CVE page)
  console.log('buildCumulativeDatasets: Checking forecast data...');
  console.log('buildCumulativeDatasets: rec.cumulative_timelines exists:', !!rec.cumulative_timelines);
  
  if (rec.cumulative_timelines) {
    console.log('buildCumulativeDatasets: Found cumulative_timelines with keys:', Object.keys(rec.cumulative_timelines));
    Object.entries(rec.cumulative_timelines).forEach(([modelKey, modelData]) => {
      console.log('buildCumulativeDatasets: Processing model:', modelKey, 'Data length:', Array.isArray(modelData) ? modelData.length : 'not array');
      
      if (Array.isArray(modelData) && modelData.length > 0) {
        const forecastData = modelData
          .filter(item => {
            const dateStr = item.date.substring(0, 4); // Extract year as string
            const isSelectedYear = parseInt(dateStr) === year;
            // Exclude year boundary reset markers (Dec 31 with value 0, which belong to next year's view)
            const isResetMarker = item.date.includes('-12-31T23:59:59Z') && item.cumulative_total === 0;
            const entryDateStr = item.date;
            // Only show forecast points AFTER last actual month (matches main CVE page)
            // This prevents duplicate display of overlap point
            const isAfterActuals = !lastActualMonthDateStr || entryDateStr > lastActualMonthDateStr;
            return isSelectedYear && !isResetMarker && isAfterActuals;
          })
          .map(item => ({
            x: new Date(item.date),
            y: item.cumulative_total
          }));
        
        console.log(`📈 Forecast data for ${modelKey} (${year}):`, forecastData.length, 'points');
        if (forecastData.length > 0) {
          console.log('  First:', forecastData[0]);
          console.log('  Last:', forecastData[forecastData.length - 1]);
          
          // Extract model name from key (remove _cumulative suffix)
          const modelName = modelKey.replace('_cumulative', '');
          
          // Get model-specific color
          const modelColor = getModelColor(modelName, 1);
          const modelColorLight = getModelColor(modelName, 0.1);
          
          const forecastDataset = {
            label: `${modelName} Forecast`,
            data: forecastData,
            borderColor: modelColor,
            backgroundColor: modelColorLight,
            borderWidth: 2,
            pointBackgroundColor: forecastData.map(point => {
              return modelColor;
            }),
            pointBorderColor: forecastData.map(point => {
              return modelColor;
            }),
            pointRadius: 3, // Same size for all forecast points
            borderDash: [5, 5],
            tension: 0.1,
            fill: false
          };
          datasets.push(forecastDataset);
        }
      }
    });
  }
  
  return datasets;
}

function generateCumulativeData(rec, year) {
  const historical = [];
  const forecasts = [];
  
  if (year === 2025) {
    // Historical data - use pre-calculated cumulative data
    if (rec.historical_cumulative && Array.isArray(rec.historical_cumulative)) {
      rec.historical_cumulative.forEach(point => {
        const date = new Date(point.date);
        if (date.getFullYear() <= 2025) {
          historical.push({
            x: date,
            y: point.cumulative_total
          });
        }
      });
    }
    
    // Forecast data for 2025
    if (rec.forecast_cumulative && rec.model_selection) {
      const selectedModel = rec.model_selection.selected_model;
      const modelData = rec.forecast_cumulative[selectedModel];
      
      if (modelData && Array.isArray(modelData)) {
        modelData.forEach(point => {
          const date = new Date(point.date);
          if (date.getFullYear() === 2025) {
            forecasts.push({
              x: date,
              y: point.cumulative_total
            });
          }
        });
      }
    }
  } else if (year === 2026) {
    // 2026 forecast data only
    if (rec.forecast_cumulative && rec.model_selection) {
      const selectedModel = rec.model_selection.selected_model;
      const modelData = rec.forecast_cumulative[selectedModel];
      
      if (modelData && Array.isArray(modelData)) {
        modelData.forEach(point => {
          const date = new Date(point.date);
          if (date.getFullYear() === 2026) {
            forecasts.push({
              x: date,
              y: point.cumulative_total
            });
          }
        });
      }
    }
  }
  
  return { historical, forecasts };
}

function renderChart(rec) {
  console.log('renderChart: Starting for CNA:', rec.name || rec.id);
  const chartSection = document.getElementById('chartSection');
  if (!chartSection) {
    console.log('renderChart: chartSection not found');
    return;
  }
  
  // Hide loading state and show chart
  const loadingState = document.getElementById('loadingState');
  if (loadingState) {
    loadingState.classList.add('hidden');
  }
  
  chartSection.classList.remove('hidden');
  
  // Show year toggle
  const yearToggle = document.getElementById('yearToggleContainer');
  if (yearToggle) {
    yearToggle.classList.remove('hidden');
  }
  
  // Get or recreate canvas
  let canvas = document.getElementById('cnaChart');
  if (!canvas) {
    return;
  }
  
  const ctx = canvas.getContext('2d');
  console.log('renderChart: Building datasets for:', rec.name || rec.id, 'Year:', currentYear);
  const datasets = buildCumulativeDatasets(rec, currentYear);
  console.log('renderChart: Built', datasets.length, 'datasets');

  if (datasets.length === 0) {
    console.log('renderChart: No datasets available, showing no data message');
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = '16px Inter';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.fillText('No data available for this CNA', ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }

  const chartData = { datasets };
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    layout: {
      padding: {
        left: 10,
        right: 10,
        top: 10,
        bottom: 10
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'month',
          tooltipFormat: 'MMM yyyy',
          displayFormats: {
            month: 'MMM'
          }
        },
        title: {
          display: true,
          text: 'Month'
        },
        ticks: {
          maxTicksLimit: 15,
          padding: 5,
          callback: function(value, index, values) {
            const date = new Date(value);
            const month = date.toLocaleDateString('en-US', { month: 'short' });
            const year = date.getFullYear();
            // Show year for January or if it's a different year
            if (date.getMonth() === 0 || (index > 0 && new Date(values[index-1].value).getFullYear() !== year)) {
              return `${month} ${year}`;
            }
            return month;
          }
        },
        min: `${currentYear}-01-01`,
        max: `${currentYear + 1}-01-05`
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Cumulative CVE Count'
        },
        ticks: {
          padding: 5,
          callback: function(value) {
            return numberFmt.format(value);
          }
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20
        }
      },
      tooltip: {
        mode: 'nearest',
        intersect: false,
        callbacks: {
          title: function(context) {
            if (context && context.length > 0) {
              const timestamp = context[0].parsed.x;
              const date = new Date(timestamp);

              // Use UTC methods to avoid timezone issues (match main CVE page)
              const month = date.getUTCMonth();
              const day = date.getUTCDate();
              const hour = date.getUTCHours();
              const minute = date.getUTCMinutes();
              const second = date.getUTCSeconds();

              // Special case for December 31st 23:59:59 - show "End of Year"
              if (month === 11 && day === 31 && hour === 23 && minute === 59 && second === 59) {
                return `End of Year ${date.getUTCFullYear()}`;
              }

              // Check if it's the last point (MTD) - show full date
              const isLastPoint = context[0].datasetIndex === 0 &&
                                 context[0].dataIndex === context[0].dataset.data.length - 1;

              if (isLastPoint && day > 1 && day < 28) {
                return date.toLocaleDateString('en-US', {
                  month: 'long',
                  day: 'numeric',
                  year: 'numeric',
                  timeZone: 'UTC'
                });
              }

              // For regular cumulative data points, show just the month name
              return date.toLocaleDateString('en-US', {
                month: 'long',
                year: 'numeric',
                timeZone: 'UTC'
              });
            }
            return '';
          },
          label: function(context) {
            const value = numberFmt.format(context.parsed.y);
            const datasetLabel = context.dataset.label;
            return `${datasetLabel}: ${value}`;
          }
        }
      }
    }
  };

  setTimeout(() => {
    try {
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
    } catch (error) {
      if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
      }
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.font = '16px Inter';
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText('Error loading chart', ctx.canvas.width / 2, ctx.canvas.height / 2);
    }
  }, 150);
}

function updateYearToggleUI() {
  const btn2025 = document.getElementById('yearCurrentBtn');
  const btn2026 = document.getElementById('yearNextBtn');
  
  const active = 'px-3 py-1 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors';
  const idle = 'px-3 py-1 text-sm font-medium bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors';
  const showingCurrent = currentYear === YEARS.currentYear;
  if (btn2025) btn2025.className = showingCurrent ? active : idle;
  if (btn2026) btn2026.className = showingCurrent ? idle : active;
}

// =============================================================================
// SUMMARY FUNCTIONS
// =============================================================================

function updateSummary(rec) {
  const displayName = getCnaDisplayName(rec.id, rec.name);
  const shortName = getCnaShortName(rec.id, rec.name);
  
  // Calculate metrics using the same logic as table
  const metrics = calculateCnaMetrics(rec);
  
  // Update CNA info card
  document.getElementById('summaryPanelTitle').textContent = displayName;
  document.getElementById('summaryId').textContent = shortName;
  
  // Update 2024 card
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };

  setText('summaryPriorYear', numberFmt.format(metrics.priorTotal));
  setText('summaryCurrentYear', numberFmt.format(metrics.forecastedCurrent));
  // Say plainly how much of the projection is already fact.
  setText(
    'summaryCurrentYearSplit',
    `${numberFmt.format(metrics.currentPublished)} published + ${numberFmt.format(metrics.currentRemainder)} forecast`
  );

  const modelInfo = document.getElementById('summaryModelInfo');
  if (modelInfo) {
    // Model choice is cached and refreshed periodically, so say when it was made
    // rather than implying it was decided on today's data.
    const chosen = cnaData?.[metrics.id]?.model_selection?.selected_at;
    const when = chosen ? ` · chosen ${new Date(chosen).toLocaleDateString()}` : '';
    const mase = typeof metrics.mase === 'number' ? ` (MASE ${metrics.mase.toFixed(2)})` : '';
    modelInfo.textContent = `${metrics.model || 'Unknown'}${mase}${when}`;
  }

  const growthRateElement = document.getElementById('summaryGrowthRate');
  if (growthRateElement) {
    const growthRate = metrics.growthRate;
    growthRateElement.textContent = `${growthRate > 0 ? '+' : ''}${growthRate.toFixed(1)}%`;

    // Recolour the whole card through the stat-card variants, rather than
    // swapping Tailwind gradient utilities that are only half-themed for dark mode.
    const card = growthRateElement.closest('.stat-card');
    if (card) {
      card.classList.remove('stat-card--green', 'stat-card--bad', 'stat-card--neutral', 'stat-card--purple');
      card.classList.add(growthRate > 0 ? 'stat-card--green' : growthRate < 0 ? 'stat-card--amber' : 'stat-card--neutral');
    }
  }

  // Update page title
  const titleText = `${displayName} - CVE Forecast`;
  const panelTitleElement = document.getElementById('panelTitle');
  if (panelTitleElement) {
    panelTitleElement.textContent = titleText;
  }
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function setYear(year) {
  currentYear = year;
  updateYearToggleUI();
  if (currentCnaData) {
    renderChart(currentCnaData);
  }
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  const btn2025 = document.getElementById('yearCurrentBtn');
  const btn2026 = document.getElementById('yearNextBtn');
  
  if (btn2025) {
    btn2025.addEventListener('click', () => setYear(YEARS.currentYear));
  }
  if (btn2026) {
    btn2026.addEventListener('click', () => setYear(YEARS.nextYear));
  }
  
  // Initialize the application
  loadCnaData();
});

// =============================================================================
// MODEL STATISTICS
// =============================================================================

function updateModelStatistics() {
  console.log('=== updateModelStatistics: STARTING ===');
  console.log('updateModelStatistics: Current time:', new Date().toISOString());
  
  // Check cnaData availability
  if (!cnaData) {
    console.log('updateModelStatistics: cnaData is null/undefined');
    return;
  }
  
  const cnaKeys = Object.keys(cnaData);
  if (cnaKeys.length === 0) {
    console.log('updateModelStatistics: cnaData is empty object');
    return;
  }
  
  console.log('updateModelStatistics: Processing', cnaKeys.length, 'CNAs');
  console.log('updateModelStatistics: First CNA ID:', cnaKeys[0]);
  console.log('updateModelStatistics: Sample CNA has model_selection:', !!cnaData[cnaKeys[0]].model_selection);

  // Check if DOM elements exist with detailed logging
  console.log('updateModelStatistics: Checking DOM elements...');
  const totalCnasEl = document.getElementById('totalCnas');
  const modelsUsedEl = document.getElementById('modelsUsed');
  const averageMapeEl = document.getElementById('averageMape');
  const bestMapeEl = document.getElementById('bestMape');
  const distributionEl = document.getElementById('modelDistribution');

  console.log('updateModelStatistics: DOM element check results:');
  console.log('  - totalCnas:', !!totalCnasEl, totalCnasEl ? 'FOUND' : 'NOT FOUND');
  console.log('  - modelsUsed:', !!modelsUsedEl, modelsUsedEl ? 'FOUND' : 'NOT FOUND');
  console.log('  - averageMape:', !!averageMapeEl, averageMapeEl ? 'FOUND' : 'NOT FOUND');
  console.log('  - bestMape:', !!bestMapeEl, bestMapeEl ? 'FOUND' : 'NOT FOUND');
  console.log('  - modelDistribution:', !!distributionEl, distributionEl ? 'FOUND' : 'NOT FOUND');

  if (!totalCnasEl || !modelsUsedEl || !averageMapeEl || !bestMapeEl || !distributionEl) {
    console.log('updateModelStatistics: Some DOM elements missing, implementing robust retry...');
    retryUpdateModelStatistics(1);
    return;
  }
  
  console.log('updateModelStatistics: All DOM elements found, proceeding with data processing...');

  // Calculate model statistics with detailed logging
  console.log('updateModelStatistics: Starting data processing...');
  const modelCounts = {};
  const mapeScores = [];
  let totalCnas = 0;
  let cnasWithModelSelection = 0;

  Object.values(cnaData).forEach((cna, index) => {
    if (cna.model_selection) {
      cnasWithModelSelection++;
      const model = cna.model_selection.selected_model;
      const mase = cna.model_selection.validation_mase;
      
      if (index < 3) { // Log first 3 for debugging
        console.log(`updateModelStatistics: CNA ${index + 1} - Model: ${model}, MASE: ${mase}`);
      }
      
      modelCounts[model] = (modelCounts[model] || 0) + 1;
      if (typeof mase === 'number' && isFinite(mase)) {
        mapeScores.push(mase);
      }
      totalCnas++;
    }
  });
  
  console.log('updateModelStatistics: Data processing complete:');
  console.log('  - Total CNAs processed:', totalCnas);
  console.log('  - CNAs with model_selection:', cnasWithModelSelection);
  console.log('  - Model counts:', modelCounts);
  console.log('  - Valid MAPE scores:', mapeScores.length);

  // Update performance metrics with logging
  console.log('updateModelStatistics: Updating DOM elements...');
  
  totalCnasEl.textContent = totalCnas;
  console.log('updateModelStatistics: Set totalCnas to:', totalCnas);
  
  modelsUsedEl.textContent = Object.keys(modelCounts).length;
  console.log('updateModelStatistics: Set modelsUsed to:', Object.keys(modelCounts).length);
  
  if (mapeScores.length > 0) {
    // Median, not mean: a handful of CNAs score 160-240% and drag an average to
    // a number no individual CNA is anywhere near.
    const sorted = [...mapeScores].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];

    averageMapeEl.textContent = median.toFixed(2);
    bestMapeEl.textContent = sorted[0].toFixed(2);
  } else {
    averageMapeEl.textContent = '—';
    bestMapeEl.textContent = '—';
  }

  // Update model distribution bars
  distributionEl.innerHTML = '';
  console.log('updateModelStatistics: Cleared distribution container');

  // Sort models by count (descending)
  const sortedModels = Object.entries(modelCounts)
    .sort(([,a], [,b]) => b - a);

  // Define colors for each model - using Tailwind v2.2.19 compatible classes
  const modelColors = {
    'Prophet': 'bg-purple-500',
    'XGBoost': 'bg-green-500',
    'LightGBM': 'bg-yellow-500',
    'ExponentialSmoothing': 'bg-red-500',
    'LinearRegression': 'bg-indigo-500',
    'AutoARIMA': 'bg-gray-500'
  };

  sortedModels.forEach(([model, count]) => {
    const percentage = ((count / totalCnas) * 100).toFixed(1);
    const color = modelColors[model] || 'bg-gray-500';
    
    const modelBar = document.createElement('div');
    modelBar.className = 'flex items-center justify-between';
    modelBar.innerHTML = `
      <div class="flex items-center space-x-3 flex-1">
        <div class="w-3 h-3 rounded-full ${color}"></div>
        <span class="text-sm font-medium text-gray-700">${model}</span>
      </div>
      <div class="flex items-center space-x-2">
        <div class="w-24 bg-gray-200 rounded-full h-2">
          <div class="${color} h-2 rounded-full" style="width: ${percentage}%"></div>
        </div>
        <span class="text-sm text-gray-600 w-12 text-right">${count}</span>
        <span class="text-xs text-gray-500 w-10 text-right">(${percentage}%)</span>
      </div>
    `;
    
    distributionEl.appendChild(modelBar);
    console.log(`updateModelStatistics: Added bar for ${model}: ${count} (${percentage}%)`);
  });
  
  console.log('updateModelStatistics: Successfully completed all updates');
  
  // Debug: Check if the card is actually visible
  const cardContainer = document.querySelector('.bg-white.rounded-lg.card-shadow');
  if (cardContainer) {
    const rect = cardContainer.getBoundingClientRect();
    console.log('updateModelStatistics: Model card position:', {
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      visible: rect.top < window.innerHeight && rect.bottom > 0
    });
    console.log('updateModelStatistics: Card display style:', window.getComputedStyle(cardContainer).display);
    console.log('updateModelStatistics: Card visibility:', window.getComputedStyle(cardContainer).visibility);
  } else {
    console.log('updateModelStatistics: ERROR - Model statistics card container not found in DOM!');
  }
  
  console.log('=== updateModelStatistics: FINISHED ===');
}

// Robust retry mechanism with exponential backoff
function retryUpdateModelStatistics(attempt) {
  const maxAttempts = 10;
  const baseDelay = 100;
  
  if (attempt > maxAttempts) {
    console.error('updateModelStatistics: Max retry attempts reached, giving up');
    return;
  }
  
  const delay = baseDelay * Math.pow(1.5, attempt - 1);
  console.log(`updateModelStatistics: Retry attempt ${attempt}/${maxAttempts} in ${delay}ms`);
  
  setTimeout(() => {
    console.log(`updateModelStatistics: Retry ${attempt} executing...`);
    
    // Check DOM readiness using multiple methods
    if (document.readyState !== 'complete') {
      console.log('updateModelStatistics: Document not ready, waiting...');
      retryUpdateModelStatistics(attempt + 1);
      return;
    }
    
    // Use requestAnimationFrame to ensure DOM is painted
    requestAnimationFrame(() => {
      console.log('updateModelStatistics: requestAnimationFrame fired, checking elements...');
      updateModelStatistics();
    });
  }, delay);
}

// Enhanced DOM ready detection
function ensureDOMReady(callback) {
  if (document.readyState === 'complete') {
    console.log('ensureDOMReady: Document already complete');
    requestAnimationFrame(callback);
    return;
  }
  
  if (document.readyState === 'interactive') {
    console.log('ensureDOMReady: Document interactive, waiting for complete');
    window.addEventListener('load', () => {
      console.log('ensureDOMReady: Window load event fired');
      requestAnimationFrame(callback);
    });
    return;
  }
  
  console.log('ensureDOMReady: Document loading, waiting for DOMContentLoaded');
  document.addEventListener('DOMContentLoaded', () => {
    console.log('ensureDOMReady: DOMContentLoaded fired');
    window.addEventListener('load', () => {
      console.log('ensureDOMReady: Window load event fired after DOMContentLoaded');
      requestAnimationFrame(callback);
    });
  });
}
