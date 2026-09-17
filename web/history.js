/**
 * Forecast history page.
 *
 * Renders web/forecast_history.json: the record of what each run predicted, how
 * much those predictions moved between runs, and how they compared to the
 * outcome once a month closed.
 *
 * The tracker was silently broken from v0.10 to v0.12 (a schema key mismatch
 * swallowed by a bare except), so this file has to cope with a history that is
 * still filling up. Empty states say what is missing and when it will appear,
 * rather than rendering an empty chart.
 */

const CACHE_BUSTER = new Date().getTime();
const ENSEMBLE = 'Ensemble';

let history = null;
let vintages = null;
let evolutionChart = null;

document.addEventListener('DOMContentLoaded', init);

/** Theme toggle, matching the other pages. */
(function initThemeToggle() {
    document.addEventListener('DOMContentLoaded', () => {
        const toggle = document.getElementById('themeToggle');
        const sun = document.getElementById('themeSun');
        const moon = document.getElementById('themeMoon');
        const sync = () => {
            const dark = document.documentElement.getAttribute('data-theme') === 'dark';
            sun?.classList.toggle('hidden', !dark);
            moon?.classList.toggle('hidden', dark);
        };
        sync();
        toggle?.addEventListener('click', () => {
            const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            try {
                localStorage.setItem('theme', next);
            } catch (e) {
                /* private browsing: the toggle still works for this page view */
            }
            sync();
            if (evolutionChart) renderEvolution(document.getElementById('monthSelector').value);
        });
    });
})();

/** Loads the history file and renders the page. */
async function init() {
    try {
        const [historyResponse, vintageResponse] = await Promise.all([
            fetch(`forecast_history.json?v=${CACHE_BUSTER}`),
            fetch(`data_vintages.json?v=${CACHE_BUSTER}`).catch(() => null),
        ]);

        if (!historyResponse.ok) throw new Error(`forecast_history.json returned ${historyResponse.status}`);
        history = await historyResponse.json();
        vintages = vintageResponse && vintageResponse.ok ? await vintageResponse.json() : null;

        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('content').classList.remove('hidden');

        renderSummary();
        populateMonthSelector();
        renderAccuracy();
        renderStability();
        renderVintages();
    } catch (error) {
        console.error('Failed to load forecast history:', error);
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('errorState').classList.remove('hidden');
        document.getElementById('errorMessage').textContent = error.message;
    }
}

const snapshots = () => history?.forecast_snapshots || [];

/** Top summary cards. */
function renderSummary() {
    const all = snapshots();
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    setText('snapshotCount', all.length.toLocaleString());
    if (all.length > 0) {
        const first = new Date(all[0].snapshot_date);
        const last = new Date(all[all.length - 1].snapshot_date);
        setText('snapshotRange', `${first.toLocaleDateString()} – ${last.toLocaleDateString()}`);
    } else {
        setText('snapshotRange', 'Recording starts with the next run');
    }

    setText('monthsScored', Object.keys(history.accuracy_tracking || {}).length.toLocaleString());

    const revisions = Object.values(history.stability_metrics || {})
        .map(m => m.mean_revision_pct)
        .filter(v => typeof v === 'number')
        .sort((a, b) => a - b);
    setText(
        'medianRevision',
        revisions.length ? `${revisions[Math.floor(revisions.length / 2)].toFixed(1)}%` : '—'
    );

    setText('lastUpdated', history.last_updated ? `Last updated: ${new Date(history.last_updated).toLocaleString()}` : '');
}

/** Months that appear in at least one snapshot, newest first. */
function forecastMonths() {
    const months = new Set();
    snapshots().forEach(s => Object.keys(s.forecasts || {}).forEach(m => months.add(m)));
    return [...months].sort();
}

/** Populates the target-month picker and draws the first chart. */
function populateMonthSelector() {
    const selector = document.getElementById('monthSelector');
    const months = forecastMonths();
    selector.innerHTML = '';

    months.forEach(month => {
        const option = document.createElement('option');
        option.value = month;
        option.textContent = new Date(`${month}-01T00:00:00Z`).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'long',
            timeZone: 'UTC',
        });
        selector.appendChild(option);
    });

    selector.addEventListener('change', () => renderEvolution(selector.value));
    if (months.length) renderEvolution(months[0]);
}

/** Reads a themed CSS custom property. */
const token = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * Draws how the forecast for one month changed across snapshots.
 * @param {string} month Target month, 'YYYY-MM'
 */
function renderEvolution(month) {
    const rows = snapshots()
        .filter(s => s.forecasts?.[month]?.[ENSEMBLE] !== undefined)
        .map(s => ({ x: new Date(s.snapshot_date), y: s.forecasts[month][ENSEMBLE] }));

    const caption = document.getElementById('evolutionCaption');
    const actual = history.accuracy_tracking?.[month]?.actual;

    if (rows.length < 2) {
        caption.textContent =
            `Only ${rows.length} recorded forecast for this month so far. The line appears once a month has ` +
            'been forecast by more than one run — with daily runs, within a couple of days.';
    } else {
        const drift = ((rows[rows.length - 1].y - rows[0].y) / rows[0].y) * 100;
        caption.textContent =
            `${rows.length} forecasts recorded. The prediction has moved ${drift >= 0 ? '+' : ''}${drift.toFixed(1)}% ` +
            'since it was first made' + (actual ? `, against an actual of ${actual.toLocaleString()}.` : '.');
    }

    const datasets = [
        {
            label: 'Ensemble forecast',
            data: rows,
            borderColor: token('--color-primary') || 'rgb(37, 99, 235)',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.1,
        },
    ];

    if (actual !== undefined && rows.length) {
        datasets.push({
            label: 'Actual (once published)',
            data: [
                { x: rows[0].x, y: actual },
                { x: rows[rows.length - 1].x, y: actual },
            ],
            borderColor: token('--color-forecast') || 'rgb(220, 38, 38)',
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
        });
    }

    const gridColor = token('--color-border') || '#e2e8f0';
    const textColor = token('--color-text-secondary') || '#475569';

    if (evolutionChart) evolutionChart.destroy();
    evolutionChart = new Chart(document.getElementById('evolutionChart'), {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, color: textColor } },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${Math.round(ctx.parsed.y).toLocaleString()} CVEs`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day' },
                    title: { display: true, text: 'When the forecast was made', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor },
                },
                y: {
                    title: { display: true, text: 'Forecast CVEs for the month', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor, callback: v => v.toLocaleString() },
                },
            },
        },
    });
}

/** Accuracy table for months that have both a forecast and an outcome. */
function renderAccuracy() {
    const tracking = history.accuracy_tracking || {};
    const body = document.getElementById('accuracyTable');
    const empty = document.getElementById('accuracyEmpty');
    const wrapper = document.getElementById('accuracyWrapper');
    body.innerHTML = '';

    const months = Object.keys(tracking).sort().reverse();
    if (months.length === 0) {
        empty.classList.remove('hidden');
        wrapper.classList.add('hidden');
        return;
    }
    empty.classList.add('hidden');
    wrapper.classList.remove('hidden');

    months.forEach(month => {
        const entry = tracking[month];
        // The forecast made furthest ahead is the interesting one: anyone can be
        // right about a month that has almost finished.
        const forecasts = (entry.forecasts_over_time || []).filter(f => f.model === ENSEMBLE);
        const earliest = forecasts.sort((a, b) => b.weeks_ahead - a.weeks_ahead)[0];
        if (!earliest) return;

        const absError = Math.abs(earliest.error_pct);
        const verdict = absError < 10 ? 'pill--good' : absError < 25 ? 'pill--warn' : 'pill--bad';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="font-mono">${month}</td>
            <td class="text-right font-mono">${entry.actual.toLocaleString()}</td>
            <td class="text-right font-mono">${earliest.forecast.toLocaleString()}</td>
            <td class="text-right font-mono">${earliest.error_pct >= 0 ? '+' : ''}${earliest.error_pct.toFixed(1)}%</td>
            <td class="text-right font-mono">${earliest.weeks_ahead.toFixed(0)} wk</td>
            <td class="text-center"><span class="pill ${verdict}">${entry.convergence_quality.replace('_', ' ')}</span></td>
        `;
        body.appendChild(row);
    });
}

/** Per-model revision volatility between consecutive runs. */
function renderStability() {
    const metrics = history.stability_metrics || {};
    const body = document.getElementById('stabilityTable');
    body.innerHTML = '';

    const ranked = Object.entries(metrics).sort((a, b) => b[1].stability_score - a[1].stability_score);
    if (ranked.length === 0) {
        body.innerHTML =
            '<tr><td colspan="4" class="text-gray-600 text-sm py-4">Needs at least two snapshots. Available after the next run.</td></tr>';
        return;
    }

    ranked.forEach(([model, m]) => {
        const verdict = m.stability_score > 0.9 ? 'pill--good' : m.stability_score > 0.7 ? 'pill--warn' : 'pill--bad';
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="font-medium text-gray-800">${model}</td>
            <td class="text-right font-mono">${m.mean_revision_pct.toFixed(2)}%</td>
            <td class="text-right font-mono">${m.max_revision_pct.toFixed(2)}%</td>
            <td class="text-center"><span class="pill ${verdict}">${m.stability_score.toFixed(2)}</span></td>
        `;
        body.appendChild(row);
    });
}

/** Data-vintage summary: how much month counts move after publication. */
function renderVintages() {
    const el = document.getElementById('vintageSummary');
    const factors = vintages?.revision_factors;

    if (!vintages || !factors) {
        el.textContent = 'Vintage recording starts with the next run.';
        return;
    }

    if (!factors.ready) {
        el.textContent =
            `Recording: ${factors.n_vintages ?? 0} observation(s) across ${factors.months_observed ?? 0} months. ` +
            'Revision factors need the same month observed on more than one day, so they appear after a few runs.';
        return;
    }

    const buckets = Object.entries(factors).filter(([, v]) => v && typeof v === 'object' && 'factor' in v);
    if (buckets.length === 0) {
        el.textContent = 'Recording, but no bucket has enough months yet to estimate a revision factor.';
        return;
    }

    el.innerHTML = buckets
        .map(
            ([bucket, v]) =>
                `<div class="flex justify-between py-1 border-b border-gray-200">
                     <span>First observed ${bucket.replace('days_', '').replace('_', '–')} days after month end</span>
                     <span class="font-mono font-semibold">${((v.factor - 1) * 100).toFixed(1)}% growth after first look
                         <span class="text-gray-500 font-normal">(n=${v.n})</span></span>
                 </div>`
        )
        .join('');
}
