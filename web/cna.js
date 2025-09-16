// CNA Forecast JavaScript - Consolidated
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
let sortColumn = 'forecasted2025';
let sortDirection = 'desc';

// Formatting
const numberFmt = new Intl.NumberFormat();

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Load CNA data
async function loadCnaData() {
  try {
    const response = await fetch('cna_data.json');
    
    if (response.ok) {
      const text = await response.text();
      
      try {
        cnaData = JSON.parse(text);
        
        // Sort CNAs by total historical CVE count
        sortedCnaIds = sortCnasByTotal(cnaData);
        
        // Initialize table
        initializeTable();
        
        // Set up table sorting event listeners
        setupTableSorting();
        
        // Update dynamic table headers
        updateDynamicHeaders();
        
        // Auto-select top CNA and display chart
        autoSelectTopCna();
      } catch (parseError) {
        // Handle JSON parsing errors silently in production
      }
    }
  } catch (error) {
    // Handle network errors silently in production
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
    
    let total2024 = 0;
    let total2025 = 0;
    let predicted2025 = 0;
    
    if (rec.historical) {
      Object.entries(rec.historical).forEach(([month, count]) => {
        if (typeof count !== 'number') {
          return;
        }
        if (month.startsWith('2024-')) {
          total2024 += count;
        } else if (month.startsWith('2025-')) {
          total2025 += count;
        }
      });
    }
    
    // Calculate predicted 2025 and 2026 values from forecast
    let predicted2026 = 0;
    if (rec.forecasts && rec.model_selection) {
      const selectedModel = rec.model_selection.selected_model;
      const modelForecasts = rec.forecasts[selectedModel];
      if (modelForecasts) {
        // 2025 remaining months
        const remainingMonths2025 = Object.keys(modelForecasts).filter(month => month.startsWith('2025-'));
        remainingMonths2025.forEach(month => {
          const forecastValue = modelForecasts[month] || 0;
          if (typeof forecastValue === 'number') {
            predicted2025 += forecastValue;
          }
        });
        
        // 2026 forecast months
        const months2026 = Object.keys(modelForecasts).filter(month => month.startsWith('2026-'));
        months2026.forEach(month => {
          const forecastValue = modelForecasts[month] || 0;
          if (typeof forecastValue === 'number') {
            predicted2026 += forecastValue;
          }
        });
      }
    }
    
    const growthRate = total2024 > 0 ? ((total2025 + predicted2025 - total2024) / total2024 * 100) : 0;
    
    const forecasted2025 = total2025 + Math.round(predicted2025);
    const forecasted2026 = Math.round(predicted2026);
    
    const result = {
      id: rec.id,
      name: displayName,
      shortName: shortName,
      total2024: total2024,
      forecasted2025: forecasted2025,
      forecasted2026: forecasted2026,
      growthRate: growthRate,
      model: rec.model_selection?.selected_model || 'N/A',
      mape: rec.model_selection?.validation_mape || 0
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
    
    // Auto-load MITRE's chart by default
    const mitreId = '8254265b-2729-46b6-b9e3-3dfca2d5bfca';
    const mitreRecord = cnaData[mitreId];
    if (mitreRecord) {
      selectCnaFromTable(mitreId);
    }
    
  } catch (error) {
    // Handle table initialization errors silently
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
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.total2024)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.forecasted2025)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${numberFmt.format(row.forecasted2026)}</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm ${growthClass}">${growthSymbol}${row.growthRate.toFixed(1)}%</td>
        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
          <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            ${row.model}
          </span>
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

// Set up table sorting functionality
function setupTableSorting() {
  const headers = document.querySelectorAll('#cnaTable th[data-sort]');
  headers.forEach(header => {
    header.addEventListener('click', () => {
      const column = header.getAttribute('data-sort');
      
      // Toggle sort direction
      if (sortColumn === column) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        sortColumn = column;
        sortDirection = 'asc';
      }
      
      // Update sort indicators
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
    });
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
  if (filteredData && filteredData.length > 0) {
    const topCna = filteredData[0];
    const rec = cnaData[topCna.id];
    if (rec) {
      currentCnaData = rec;
      updateSummary(rec);
      renderChart(rec);
    }
  }
}

// CHART FUNCTIONS
// =============================================================================

function getModelColor(modelName, alpha = 1) {
  const colors = {
    'Prophet': `rgba(34, 197, 94, ${alpha})`,
    'XGBoost': `rgba(239, 68, 68, ${alpha})`,
    'LightGBM': `rgba(168, 85, 247, ${alpha})`,
    'AutoARIMA': `rgba(59, 130, 246, ${alpha})`,
    'ExponentialSmoothing': `rgba(245, 158, 11, ${alpha})`,
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
    'TiDE': `rgba(245, 101, 101, ${alpha})`,
    'DLinear': `rgba(52, 211, 153, ${alpha})`
  };
  
  return colors[modelName] || `rgba(107, 114, 128, ${alpha})`;
}

function buildCumulativeDatasets(rec, year) {
  
  const datasets = [];
  
  // Historical data (actual CVEs published)
  if (rec.historical_cumulative && Array.isArray(rec.historical_cumulative)) {
    const historicalData = rec.historical_cumulative
      .filter(item => {
        const itemDate = new Date(item.date);
        const itemYear = itemDate.getFullYear();
        return itemYear === year;
      })
      .map(item => {
        const dateObj = new Date(item.date);
        return {
          x: dateObj,
          y: item.cumulative_total
        };
      });
    
    // Add current month-to-date data point if we have current month data
    const currentDate = new Date();
    const currentYear = currentDate.getFullYear();
    const currentMonth = currentDate.getMonth() + 1; // getMonth() is 0-based
    const currentMonthKey = `${currentYear}-${currentMonth.toString().padStart(2, '0')}`;
    
    if (year === currentYear && rec.historical && rec.historical[currentMonthKey]) {
      const lastHistoricalPoint = historicalData[historicalData.length - 1];
      const currentMonthCVEs = rec.historical[currentMonthKey];
      const currentMonthCumulative = lastHistoricalPoint ? lastHistoricalPoint.y + currentMonthCVEs : currentMonthCVEs;
      
      // Add current month-to-date point (mid-month)
      const currentMonthDate = new Date(currentYear, currentMonth - 1, 15, 12, 0, 0); // 15th of current month
      historicalData.push({
        x: currentMonthDate,
        y: currentMonthCumulative
      });
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
  
  // Forecast data - use cumulative_timelines structure (matching main.py)
  
  if (rec.cumulative_timelines) {
    Object.entries(rec.cumulative_timelines).forEach(([modelKey, modelData]) => {
      
      if (Array.isArray(modelData) && modelData.length > 0) {
        const forecastData = modelData
          .filter(item => {
            const itemDate = new Date(item.date);
            const itemYear = itemDate.getFullYear();
            const itemMonth = itemDate.getMonth();
            const itemDay = itemDate.getDate();
            // Include forecast data for the selected year
            const includeItem = itemYear === year;
            return includeItem;
          })
          .map(item => {
            const dateObj = new Date(item.date);
            return {
              x: dateObj,
              y: item.cumulative_total
            };
          });
        
        
        if (forecastData.length > 0) {
          // Extract model name from key (remove _cumulative suffix)
          const modelName = modelKey.replace('_cumulative', '');
          
          const forecastDataset = {
            label: `${modelName} Forecast`,
            data: forecastData,
            borderColor: '#dc2626', // Red color for all forecast lines
            backgroundColor: 'rgba(220, 38, 38, 0.1)', // Light red background
            borderWidth: 2,
            pointBackgroundColor: forecastData.map(point => {
              return '#dc2626'; // Red for all forecast points including year-end
            }),
            pointBorderColor: forecastData.map(point => {
              return '#dc2626'; // Red for all forecast points including year-end
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
  const chartSection = document.getElementById('chartSection');
  if (!chartSection) {
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
  
  // Destroy existing chart if it exists
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
  
  const ctx = canvas.getContext('2d');
  const datasets = buildCumulativeDatasets(rec, currentYear);
  
  if (datasets.length === 0) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = '16px Inter';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.fillText('No data available for this CNA', ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }
  
  setTimeout(() => {
    try {
      
      chartInstance = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
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
                    
                    // Special case for December 31st - show "End of Year 2025"
                    if (date.getMonth() === 11 && date.getDate() === 31) {
                      return `End of Year ${date.getFullYear()}`;
                    }
                    
                    // Special case for current month data point (15th of month) - show specific date
                    if (date.getDate() === 15) {
                      return date.toLocaleDateString('en-US', { 
                        year: 'numeric', 
                        month: 'long',
                        day: 'numeric'
                      });
                    }
                    
                    // For regular cumulative data points, show just the month name
                    return date.toLocaleDateString('en-US', { 
                      year: 'numeric', 
                      month: 'long'
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
        }
      });
    } catch (error) {
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.font = '16px Inter';
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText('Error loading chart', ctx.canvas.width / 2, ctx.canvas.height / 2);
    }
  }, 150);
}

function updateYearToggleUI() {
  const btn2025 = document.getElementById('year2025Btn');
  const btn2026 = document.getElementById('year2026Btn');
  
  if (currentYear === 2025) {
    btn2025.className = 'px-3 py-1 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors';
    btn2026.className = 'px-3 py-1 text-sm font-medium bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors';
  } else {
    btn2025.className = 'px-3 py-1 text-sm font-medium bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors';
    btn2026.className = 'px-3 py-1 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors';
  }
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
  document.getElementById('summary2024').textContent = numberFmt.format(metrics.total2024);
  
  // Update 2025 forecast card
  document.getElementById('summary2025').textContent = numberFmt.format(metrics.forecasted2025);
  const modelInfo = document.getElementById('summaryModelInfo');
  if (modelInfo) {
    const selectedModel = metrics.model || 'Unknown';
    const mape = metrics.mape || 0;
    modelInfo.textContent = `${selectedModel} (MAPE: ${mape.toFixed(1)}%)`;
  }
  
  // Update growth rate card with dynamic styling
  const growthRateElement = document.getElementById('summaryGrowthRate');
  if (growthRateElement) {
    const growthRate = metrics.growthRate;
    const growthSymbol = growthRate > 0 ? '+' : '';
    growthRateElement.textContent = `${growthSymbol}${growthRate.toFixed(1)}%`;
    
    // Dynamic color based on growth
    const parentCard = growthRateElement.closest('.bg-gradient-to-br');
    if (parentCard) {
      // Remove existing color classes
      parentCard.className = parentCard.className.replace(/from-\w+-\d+|to-\w+-\d+|border-\w+-\d+/g, '');
      
      if (growthRate > 0) {
        // Positive growth - green
        parentCard.classList.add('from-green-50', 'to-green-100', 'border-green-500');
        growthRateElement.classList.remove('text-red-900', 'text-gray-900');
        growthRateElement.classList.add('text-green-900');
      } else if (growthRate < 0) {
        // Negative growth - red
        parentCard.classList.add('from-red-50', 'to-red-100', 'border-red-500');
        growthRateElement.classList.remove('text-green-900', 'text-gray-900');
        growthRateElement.classList.add('text-red-900');
      } else {
        // No growth - gray
        parentCard.classList.add('from-gray-50', 'to-gray-100', 'border-gray-500');
        growthRateElement.classList.remove('text-green-900', 'text-red-900');
        growthRateElement.classList.add('text-gray-900');
      }
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
  const btn2025 = document.getElementById('year2025Btn');
  const btn2026 = document.getElementById('year2026Btn');
  
  if (btn2025) {
    btn2025.addEventListener('click', () => setYear(2025));
  }
  if (btn2026) {
    btn2026.addEventListener('click', () => setYear(2026));
  }
  
  // Initialize the application
  loadCnaData();
});
