/**
 * CVEForecast dashboard.
 *
 * Renders the overview page from data.json. Replaces script.js, which depended
 * on Tailwind utility classes being applied from the CDN at runtime.
 */

const TOP_MODELS_COUNT = 5;
const CACHE_BUSTER = Date.now();

/* Fixed hues for the comparison lines. These are series identities rather than
   theme colours, so they stay put across light and dark; the lead line and
   everything else on the chart is driven by the CSS tokens. */
const SERIES_COLORS = ['#8b5cf6', '#0ea5a4', '#f59e0b', '#ec4899', '#64748b'];

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'];

let forecastData = null;
let modelInfoData = null;
let chart = null;
let selectedYear = new Date().getFullYear();

const root = document.documentElement;
const $ = id => document.getElementById(id);

/* Write through helpers that tolerate a missing element. index.html and app.js
   are cached independently, so a visitor can briefly hold a new page with a
   stale script (or the reverse) after a deploy. Addressing an element that the
   other half has since renamed should cost that one field, not blank the whole
   dashboard behind an error card. */
const setText = (id, value) => { const el = $(id); if (el) el.textContent = value; };
const setHTML = (id, value) => { const el = $(id); if (el) el.innerHTML = value; };

const nf = new Intl.NumberFormat('en-US');
const n = v => nf.format(Math.round(v));

/* Modelled figures are rounded to the nearest thousand; counts of what has
   already been published are not. An 80% interval some 22,000 wide quoted to
   the unit claims five significant figures the model cannot support, and the
   rounding is what tells a reader which numbers are estimated and which are
   observed. Detail tables stay exact, because that is where the month-level
   errors are actually being checked. */
const approx = v => nf.format(Math.round(v / 1000) * 1000);
const approxK = v => `${Math.round(v / 1000)}k`;
const signed = v => (v > 0 ? '+' : v < 0 ? '−' : '') + nf.format(Math.abs(Math.round(v)));
const pct = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d) + '%';
const token = name => getComputedStyle(root).getPropertyValue(name).trim();
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* ==========================================================================
   Theme

   Switching lives in shell.js, shared by every page. The chart bakes token
   colours into its options when it is built and Chart.js cannot know they
   later changed, so it repaints on the event the shell emits.
   ========================================================================== */

document.addEventListener('themechange', () => restyleChart());

/* ==========================================================================
   Data
   ========================================================================== */

document.addEventListener('DOMContentLoaded', loadForecastData);

async function loadForecastData() {
    try {
        const [data, modelInfo] = await Promise.all([
            fetch(`data.json?v=${CACHE_BUSTER}`).then(r => {
                if (!r.ok) throw new Error(`data.json: ${r.status}`);
                return r.json();
            }),
            fetch(`model_info.json?v=${CACHE_BUSTER}`).then(r => (r.ok ? r.json() : null)).catch(() => null),
        ]);

        forecastData = data;
        modelInfoData = modelInfo;

        $('loadingState').hidden = true;
        $('dashboard').hidden = false;
        initializeDashboard();
    } catch (error) {
        console.error('Failed to load forecast data:', error);
        $('loadingState').hidden = true;
        $('errorState').hidden = false;
    }
}

function initializeDashboard() {
    const years = Object.keys(forecastData.yearly_forecast_totals || {})
        .map(Number)
        .filter(y => (forecastData.yearly_forecast_totals[y]?.Ensemble?.months_forecast ?? 0) > 0)
        .sort((a, b) => a - b);
    if (years.length && !years.includes(selectedYear)) selectedYear = years[0];

    renderYearButtons(years);
    renderHero();
    renderStatStrip();
    renderRankings();
    populateModelSelector();
    renderValidationTable();
    renderMethodology();
    rebuildChart();

    $('validationModelSelector').addEventListener('change', renderValidationTable);

    $('lastUpdated').textContent =
        `Updated ${new Date(forecastData.generated_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}`;
}

function renderYearButtons(years) {
    [['yearBtnCurrent', years[0]], ['yearBtnNext', years[1]]].forEach(([id, year]) => {
        const btn = $(id);
        if (!btn) return;
        if (year === undefined) { btn.hidden = true; return; }
        btn.hidden = false;
        btn.textContent = year;
        btn.dataset.year = year;
        btn.setAttribute('aria-pressed', String(year === selectedYear));
        btn.onclick = () => switchYear(year, years);
    });
}

function switchYear(year, years) {
    selectedYear = year;
    renderYearButtons(years);
    renderHero();
    rebuildChart();
}

/* ==========================================================================
   Hero
   ========================================================================== */

function renderHero() {
    const projection = forecastData.yearly_forecast_totals?.[selectedYear]?.Ensemble;
    $('heroEyebrow').textContent = `Projected CVE publications · ${selectedYear}`;

    if (!projection) {
        setText('heroRange', '—');
        setText('heroCompare', '');
        setText('heroPoint', '');
        setHTML('heroMeta', '');
        return;
    }

    /* Lead with the interval. A point estimate a year or more out claims a
       precision the model cannot deliver, so the range is the headline and the
       central estimate is a supporting line. */
    const hasBand = projection.lower_80 != null && projection.upper_80 != null;
    setHTML('heroRange', hasBand
        ? `${approx(projection.lower_80)}<span class="to">–</span>${approx(projection.upper_80)}`
        : approx(projection.total));

    setHTML('heroPoint', hasBand
        ? `80% prediction interval · central estimate <b>${approx(projection.total)}</b>`
        : 'Central estimate');

    const prev = forecastData.yearly_forecast_totals?.[selectedYear - 1]?.Ensemble;
    const prevTotal = prev?.total ?? forecastData.summary?.previous_year_total ?? 0;
    const compare = $('heroCompare');

    if (compare && prevTotal) {
        /* A multiple of last year is what a reader actually carries away, and
           it is quoted as a range for the same reason the headline is: stated
           as a single figure it would reintroduce the precision the interval
           exists to avoid. */
        const low = (hasBand ? projection.lower_80 : projection.total) / prevTotal;
        const high = (hasBand ? projection.upper_80 : projection.total) / prevTotal;

        /* The year before may itself still be part forecast — 2027 is compared
           against a 2026 that has not finished — so do not call it published. */
        const prevSettled = (prev?.months_forecast ?? 0) === 0;
        const prevLabel = prevSettled
            ? `the ${n(prevTotal)} published in ${selectedYear - 1}`
            : `the ${approx(prevTotal)} projected for ${selectedYear - 1}`;

        compare.innerHTML = hasBand
            ? `<b>${low.toFixed(1)}×</b> to <b>${high.toFixed(1)}×</b> ${prevLabel}`
            : `<b>${low.toFixed(1)}×</b> ${prevLabel}`;
        compare.className = 'figure__compare';
    } else if (compare) {
        compare.textContent = '';
        compare.className = 'figure__compare';
    }

    const parts = [];
    if (projection.months_actual > 0) {
        parts.push(`<b>${n(projection.actual_ytd)}</b> published (${projection.months_actual} mo)`);
    }
    parts.push(`<b>${approx(projection.forecast_remainder)}</b> forecast (${projection.months_forecast} mo)`);
    parts.push(`Ensemble of <b>${(forecastData.methodology?.ensemble_members || []).length}</b> models`);
    if (prevTotal) parts.push(`Prior year <b>${n(prevTotal)}</b>`);  /* settled count, exact */
    setHTML('heroMeta', parts.map(p => `<span>${p}</span>`).join(''));
}

/* ==========================================================================
   Stat strip
   ========================================================================== */

function renderStatStrip() {
    const rankings = forecastData.model_rankings || [];
    const best = rankings.find(m => !m.is_baseline);
    const scored = rankings.filter(m => !m.is_baseline && m.mase != null);
    const winners = scored.filter(m => m.beats_naive);
    const coverage = forecastData.methodology?.interval_coverage?.['80'];

    const cells = [
        {
            label: 'Leading model',
            value: best ? best.model_name : '—',
            note: best ? `Lowest MASE over ${best.n_origins} origins` : '',
        },
        {
            label: 'Accuracy (MASE)',
            value: best?.mase != null ? best.mase.toFixed(2) : '—',
            note: best?.naive_threshold
                ? `Naive baseline ${best.naive_threshold.toFixed(2)} · lower is better`
                : 'Lower is better',
        },
        {
            label: 'Beating the baseline',
            value: `${winners.length} of ${scored.length}`,
            note: 'Models that earn their place',
        },
        {
            label: '80% interval coverage',
            value: coverage ? `${(coverage.empirical * 100).toFixed(1)}%` : '—',
            note: coverage ? `Nominal 80% · n=${n(coverage.n)}` : 'Not yet available',
        },
    ];

    $('statStrip').innerHTML = cells.map(c => `
        <div class="stat">
            <p class="eyebrow">${c.label}</p>
            <p class="stat__value">${esc(c.value)}</p>
            <p class="stat__note">${c.note}</p>
        </div>`).join('');
}

/* ==========================================================================
   Model rankings
   ========================================================================== */

function renderRankings() {
    const body = $('modelRankingsTable');
    const rankings = forecastData.model_rankings || [];

    const basis = $('rankingBasis');
    if (basis && rankings[0]) {
        basis.textContent = `${rankings[0].n_origins} rolling origins · lower MASE is better`;
    }

    body.innerHTML = '';

    rankings.forEach((model, index) => {
        let pillClass, verdict;
        if (model.is_baseline)      { pillClass = 'pill--neutral'; verdict = 'Baseline'; }
        else if (model.mase == null) { pillClass = 'pill--bad';     verdict = 'Failed'; }
        else if (model.beats_naive)  { pillClass = 'pill--good';    verdict = 'Beats naive'; }
        else                         { pillClass = 'pill--warn';    verdict = 'Loses to naive'; }

        const url = modelInfoData ? modelInfoData[model.model_name] : null;
        const nameHtml = url
            ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(model.model_name)}</a>`
            : esc(model.model_name);
        const ensembleTag = model.in_ensemble
            ? '<span class="in-ensemble" title="Included in the published ensemble">●</span>'
            : '';

        const fmt = (v, digits, suffix = '') => (v == null ? '—' : v.toFixed(digits) + suffix);
        const spread = model.mase_std != null ? `<span class="spread">±${model.mase_std.toFixed(2)}</span>` : '';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="rank">${index + 1}</td>
            <td><span class="model-name">${nameHtml}${ensembleTag}</span></td>
            <td class="num strong">${fmt(model.mase, 2)}${spread}</td>
            <td class="num">${fmt(model.mape, 1, '%')}</td>
            <td class="num">${model.bias_pct == null ? '—' : pct(model.bias_pct)}</td>
            <td><span class="pill ${pillClass}">${verdict}</span></td>
            <td class="center">
                <button class="expand-btn" id="expandBtn${index}" aria-expanded="false"
                        aria-controls="configRow${index}"
                        aria-label="Show configuration for ${esc(model.model_name)}">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"></path>
                    </svg>
                </button>
            </td>`;
        row.querySelector('.expand-btn').addEventListener('click', () => toggleConfig(index));
        body.appendChild(row);

        const configRow = document.createElement('tr');
        configRow.id = `configRow${index}`;
        configRow.className = 'config-row hidden-row';
        configRow.innerHTML = `
            <td colspan="7">
                <div class="config-row__head">
                    <h4>Configuration for ${esc(model.model_name)}</h4>
                    <button type="button" class="btn" id="copyBtn${index}">Copy JSON</button>
                </div>
                <div class="config-display" id="configDisplay${index}"></div>
            </td>`;
        configRow.querySelector('.btn').addEventListener('click', () => copyConfig(index));
        body.appendChild(configRow);

        $(`configDisplay${index}`).innerHTML = formatModelConfig(model);
    });
}

function toggleConfig(index) {
    const row = $(`configRow${index}`);
    const btn = $(`expandBtn${index}`);
    const nowHidden = row.classList.toggle('hidden-row');
    btn.setAttribute('aria-expanded', String(!nowHidden));
}

async function copyConfig(index) {
    const config = formatModelConfigForDarts(forecastData.model_rankings[index]);
    const btn = $(`copyBtn${index}`);
    const restore = () => setTimeout(() => { btn.textContent = 'Copy JSON'; }, 2000);

    try {
        await navigator.clipboard.writeText(config);
        btn.textContent = 'Copied';
        restore();
    } catch (error) {
        const area = document.createElement('textarea');
        area.value = config;
        document.body.appendChild(area);
        area.select();
        try { document.execCommand('copy'); btn.textContent = 'Copied'; }
        catch (e) { btn.textContent = 'Copy failed'; }
        document.body.removeChild(area);
        restore();
    }
}

function cleanHyperparameters(model) {
    const cleaned = {};
    for (const [key, value] of Object.entries(model.hyperparameters || {})) {
        if (value !== null && value !== undefined) cleaned[key] = value;
    }
    return cleaned;
}

function formatModelConfig(model) {
    const hp = cleanHyperparameters(model);
    const config = { model_name: model.model_name };

    if (Object.keys(hp).length) config.hyperparameters = hp;
    config.split_ratio = parseFloat((model.split_ratio || 0.8).toFixed(3));
    if (model.tuned_at) config.tuned_at = model.tuned_at;

    config.performance_metrics = {
        mape: parseFloat((model.mape || 0).toFixed(4)),
        mae: parseFloat((model.mae || 0).toFixed(4)),
    };
    if (model.training_time != null) {
        config.performance_metrics.training_time = parseFloat(model.training_time.toFixed(6));
    }
    if (!Object.keys(hp).length) {
        config.note = 'Run comprehensive tuner to generate optimal hyperparameters';
    }

    return syntaxHighlightJSON(JSON.stringify(config, null, 2).trim());
}

function formatModelConfigForDarts(model) {
    const hp = cleanHyperparameters(model);
    const config = { model: model.model_name };

    if (Object.keys(hp).length) config.hyperparameters = hp;
    config.split_ratio = parseFloat((model.split_ratio || 0.8).toFixed(3));
    config.expected_performance = {
        mape: parseFloat((model.mape || 0).toFixed(4)),
        mae: parseFloat((model.mae || 0).toFixed(4)),
    };
    if (model.training_time != null) {
        config.expected_performance.training_time = parseFloat(model.training_time.toFixed(6));
    }
    if (!Object.keys(hp).length) {
        config.note = 'No hyperparameters available - run comprehensive tuner for optimal parameters';
    }

    return JSON.stringify(config, null, 2);
}

function syntaxHighlightJSON(json) {
    return esc(json).replace(
        /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        match => {
            let cls = 'json-number';
            if (/^"/.test(match)) cls = /:$/.test(match) ? 'json-key' : 'json-string';
            else if (/true|false/.test(match)) cls = 'json-boolean';
            else if (/null/.test(match)) cls = 'json-null';
            return `<span class="${cls}">${match}</span>`;
        });
}

/* ==========================================================================
   Forecast vs published
   ========================================================================== */

const GRADE_PILL = { Excellent: 'good', Good: 'info', Fair: 'warn', Poor: 'bad' };

function populateModelSelector() {
    const selector = $('validationModelSelector');
    const available = forecastData.forecast_vs_published || {};
    const models = (forecastData.model_rankings || [])
        .filter(m => !m.is_baseline && available[m.model_name])
        .slice(0, TOP_MODELS_COUNT);

    selector.innerHTML = models
        .map(m => `<option value="${esc(m.model_name)}">${esc(m.model_name)}</option>`)
        .join('');
}

function renderValidationTable() {
    const body = $('validationTable');
    const summary = $('validationSummary');
    const model = $('validationModelSelector').value;
    const modelData = forecastData.forecast_vs_published?.[model];

    if (!modelData) {
        body.innerHTML = '<tr><td colspan="6" class="center">No data available for this model.</td></tr>';
        summary.innerHTML = '';
        return;
    }

    const stats = modelData.summary_stats || {};
    summary.innerHTML = [
        `<span>MAE <b>${n(stats.mean_absolute_error || 0)}</b></span>`,
        `<span>MAPE <b>${(stats.mean_absolute_percentage_error || 0).toFixed(2)}%</b></span>`,
        `<span><b>${modelData.table_data.length}</b> months scored</span>`,
    ].join('');

    /* Group by year so a multi-year table stays readable. Older years start
       collapsed; the current one is the reason anybody is looking. */
    const byYear = {};
    modelData.table_data.forEach(row => {
        const year = row.MONTH.slice(0, 4);
        (byYear[year] ||= []).push(row);
    });

    const currentYear = String(new Date().getFullYear());
    const years = Object.keys(byYear).sort((a, b) => b - a);
    const multiYear = years.length > 1;

    body.innerHTML = '';

    years.forEach(year => {
        const collapsed = multiYear && year !== currentYear;
        const months = byYear[year].sort((a, b) => b.MONTH.localeCompare(a.MONTH));

        if (multiYear) {
            const header = document.createElement('tr');
            header.className = 'year-header';
            header.innerHTML = `<td colspan="6" class="strong" style="cursor:pointer">
                <span class="toggle-icon">${collapsed ? '▶' : '▼'}</span> ${year}</td>`;
            header.addEventListener('click', () => {
                const nowCollapsed = !header.querySelector('.toggle-icon').textContent.includes('▶');
                header.querySelector('.toggle-icon').textContent = nowCollapsed ? '▶' : '▼';
                body.querySelectorAll(`[data-year="${year}"]`)
                    .forEach(r => r.classList.toggle('hidden-row', nowCollapsed));
            });
            body.appendChild(header);
        }

        months.forEach(row => {
            const tr = document.createElement('tr');
            tr.dataset.year = year;
            if (collapsed) tr.classList.add('hidden-row');
            const month = MONTHS[Number(row.MONTH.slice(5, 7)) - 1];
            tr.innerHTML = `
                <td class="strong">${month}</td>
                <td class="num strong">${n(row.PUBLISHED)}</td>
                <td class="num">${n(row.FORECAST)}</td>
                <td class="num">${signed(row.ERROR)}</td>
                <td class="num">${pct(row.PERCENT_ERROR, 2)}</td>
                <td><span class="pill pill--${GRADE_PILL[row.PERFORMANCE] || 'neutral'}">${esc(row.PERFORMANCE)}</span></td>`;
            body.appendChild(tr);
        });
    });
}

/* ==========================================================================
   Methodology
   ========================================================================== */

function renderMethodology() {
    const meta = forecastData.methodology;
    if (!meta) return;

    const rankings = forecastData.model_rankings || [];
    const scored = rankings.filter(m => !m.is_baseline && m.mase != null);
    const winners = scored.filter(m => m.beats_naive);
    const coverage = meta.interval_coverage || {};

    const coverageRows = Object.entries(coverage).map(([level, stats]) =>
        `<dt>${level}% interval</dt>
         <dd class="${stats.calibrated ? 'pos' : 'neg'}">${(stats.empirical * 100).toFixed(1)}%</dd>`).join('');

    $('methodologyPanel').innerHTML = `
        <div>
            <p class="eyebrow">Against the baseline</p>
            <dl>
                <dt>Models beating naive</dt><dd>${winners.length} of ${scored.length}</dd>
                <dt>Naive MASE</dt><dd>${meta.naive_threshold?.toFixed(2) ?? '—'}</dd>
            </dl>
            <p>The rest are scored and published, but not used.</p>
        </div>
        <div>
            <p class="eyebrow">Interval calibration</p>
            ${coverageRows ? `<dl>${coverageRows}</dl>` : '<p>Not yet available.</p>'}
            <p>How often the published band actually contained the outcome, in backtest.</p>
        </div>
        <div>
            <p class="eyebrow">Ensemble</p>
            <dl><dt>Members</dt><dd>${(meta.ensemble_members || []).length}</dd></dl>
            <p>${esc((meta.ensemble_members || []).join(' · ')) || 'none'} — trimmed mean over the models that cleared the baseline.</p>
        </div>`;

    const s = meta.settings || {};
    $('methodologySettings').textContent = [
        s.log_space ? 'log-space' : 'levels',
        s.business_day_normalise ? 'business-day normalised' : 'raw monthly',
        `trend damping φ=${s.damping_phi}`,
        `training window ${s.training_window_months ? s.training_window_months + ' mo' : 'full history'}`,
        `ranking metric ${meta.ranking_metric}`,
    ].join(' · ');
}

/* ==========================================================================
   Chart
   ========================================================================== */

const toXY = rows => rows.map(p => ({ x: new Date(p.date), y: p.cumulative_total }));

/* Marks where the record stops and the model starts, and labels the lead
   line's endpoint so the headline figure appears on the chart rather than
   only in a tooltip you have to go looking for. */
const annotate = {
    id: 'annotate',
    afterDatasetsDraw(c) {
        const { ctx, chartArea: area, scales } = c;

        if (c.$cutover) {
            const x = scales.x.getPixelForValue(c.$cutover);
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.strokeStyle = token('--border-strong');
            ctx.beginPath();
            ctx.moveTo(x, area.top);
            ctx.lineTo(x, area.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = '600 10px Inter, sans-serif';
            ctx.fillStyle = token('--text-subtle');
            ctx.textAlign = 'left';
            ctx.fillText('TODAY', x + 6, area.top + 11);
            ctx.restore();
        }

        /* Close the forecast with the interval rather than a single figure:
           a bracket spanning the 80% range, with both bounds labelled. The
           central estimate gets a tick and no label — it is on the page, in
           the hero, where it is explicitly marked as the central estimate. */
        const lastValue = ds => {
            let i = ds.data.length - 1;
            while (i >= 0 && ds.data[i].y == null) i--;
            return i;
        };

        const upperIdx = c.data.datasets.findIndex(d => d.$bandUpper);
        const lowerIdx = c.data.datasets.findIndex(d => d.$bandLower);

        if (upperIdx >= 0 && lowerIdx >= 0) {
            const upper = c.data.datasets[upperIdx];
            const lower = c.data.datasets[lowerIdx];
            const k = lastValue(upper);
            const upperPt = c.getDatasetMeta(upperIdx).data[k];
            const lowerPt = c.getDatasetMeta(lowerIdx).data[k];

            if (k >= 0 && upperPt && lowerPt) {
                const x = upperPt.x;
                ctx.save();
                ctx.strokeStyle = token('--band-edge');
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(x, upperPt.y);
                ctx.lineTo(x, lowerPt.y);
                ctx.moveTo(x - 4, upperPt.y);
                ctx.lineTo(x + 4, upperPt.y);
                ctx.moveTo(x - 4, lowerPt.y);
                ctx.lineTo(x + 4, lowerPt.y);
                ctx.stroke();

                ctx.font = '600 11.5px Inter, sans-serif';
                ctx.fillStyle = token('--accent');
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(approxK(upper.data[k].y), x + 8, upperPt.y);
                ctx.fillText(approxK(lower.data[k].y), x + 8, lowerPt.y);
                ctx.restore();
            }
        }

        const idx = c.data.datasets.findIndex(d => d.$primary);
        if (idx < 0 || !c.isDatasetVisible(idx)) return;
        const ds = c.data.datasets[idx];
        const k = lastValue(ds);
        const point = c.getDatasetMeta(idx).data[k];
        if (k < 0 || !point) return;

        ctx.save();
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = token('--accent');
        ctx.fill();
        ctx.restore();
    },
};

function buildDatasets() {
    const { actuals_cumulative = [], cumulative_timelines = {}, cumulative_band } = forecastData;

    /* Year-boundary reset markers belong to the next year's view, not this one. */
    const inYear = rows => rows.filter(d => {
        const isReset = d.date.includes('-12-31T23:59:59Z') && d.cumulative_total === 0;
        return Number(d.date.slice(0, 4)) === selectedYear && !isReset;
    });

    const actualPts = inYear(actuals_cumulative);
    const cutover = actualPts.length ? new Date(actualPts[actualPts.length - 1].date) : null;

    /* Everything forecast begins where the actuals end, so the page never
       draws a prediction across months that already have published counts,
       and the interval band correctly has zero width on its first day. */
    const fromCutover = rows => {
        const pts = inYear(rows);
        return cutover ? pts.filter(d => new Date(d.date) >= cutover) : pts;
    };

    const sets = [];

    if (cumulative_band?.lower?.length && cumulative_band?.upper?.length) {
        const lower = toXY(fromCutover(cumulative_band.lower));
        const upper = toXY(fromCutover(cumulative_band.upper));
        if (lower.length > 1 && upper.length > 1) {
            /* Edges drawn, not just a fill: the interval is the headline claim,
               so it needs to read as a shape rather than a faint wash. */
            sets.push({ label: '80% range', pts: upper, borderWidth: 1, pointRadius: 0,
                        borderColor: token('--band-edge'), fill: '+1',
                        backgroundColor: token('--band'),
                        $band: true, $bandUpper: true, order: 10 });
            sets.push({ label: '_lower', pts: lower, borderWidth: 1, pointRadius: 0,
                        borderColor: token('--band-edge'),
                        $band: true, $bandLower: true, order: 10 });
        }
    }

    if (actualPts.length) {
        sets.push({ label: 'Published', pts: toXY(actualPts), borderColor: token('--accent'),
                    borderWidth: 2.25, pointRadius: 0, tension: .25, order: 1, $themed: true });
    }

    /* Lead with the ensemble: it is what the headline figure and the band both
       describe, so the number, the range and the line all agree. */
    const ranked = (forecastData.model_rankings || [])
        .filter(m => !m.is_baseline && cumulative_timelines[`${m.model_name}_cumulative`])
        .slice(0, TOP_MODELS_COUNT)
        .map(m => m.model_name);
    const names = ['Ensemble', ...ranked.filter(name => name !== 'Ensemble')];

    names.forEach((name, i) => {
        const timeline = cumulative_timelines[`${name}_cumulative`];
        if (!timeline) return;
        const pts = toXY(fromCutover(timeline));
        if (pts.length < 2) return;

        const primary = name === 'Ensemble';
        sets.push({
            label: name,
            pts,
            borderColor: primary ? token('--accent') : SERIES_COLORS[(i - 1) % SERIES_COLORS.length],
            borderWidth: primary ? 1.75 : 1.25,
            borderDash: [5, 4],
            pointRadius: 0,
            tension: .25,
            $primary: primary,
            $themed: primary,
            order: primary ? 2 : 5,
            hidden: !primary,
        });
    });

    /* Put every series on one shared date grid, with null where a series has
       no value. Chart.js's index interaction matches datasets by ARRAY
       POSITION, not by date: the actuals run monthly from January while the
       forecasts only start at the cutover, so without this a hover on
       February would read Published[2] alongside each model's [2] — a
       November figure under a February heading. */
    const grid = [...new Set(sets.flatMap(s => s.pts.map(p => +p.x)))].sort((a, b) => a - b);
    sets.forEach(s => {
        const byTime = new Map(s.pts.map(p => [+p.x, p.y]));
        s.data = grid.map(t => ({ x: t, y: byTime.has(t) ? byTime.get(t) : null }));
        delete s.pts;
    });

    return { sets, cutover };
}

function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        /* Room at the right edge for the interval bracket's labels. */
        layout: { padding: { right: 62 } },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: token('--surface'),
                borderColor: token('--border'),
                borderWidth: 1,
                titleColor: token('--text'),
                bodyColor: token('--text-muted'),
                padding: 10,
                cornerRadius: 6,
                boxPadding: 4,
                usePointStyle: true,
                /* Drop the band, and any series with no value on this date —
                   a forecast before the cutover, or the actuals after it. */
                filter: item => !item.dataset.$band && item.parsed.y != null,
                callbacks: {
                    title: items => {
                        const d = new Date(items[0].parsed.x);
                        /* The as-of-today sample shares a month with the month
                           start, so name the day when it is not the 1st. */
                        return d.getUTCDate() === 1
                            ? d.toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })
                            : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
                    },
                    label: item => ` ${item.dataset.label}  ${n(item.parsed.y)}`,
                },
            },
        },
        scales: {
            x: {
                type: 'time',
                time: { unit: 'month' },
                grid: { display: false },
                border: { color: token('--border') },
                ticks: { color: token('--text-subtle'), font: { size: 11 }, maxRotation: 0, autoSkipPadding: 20 },
            },
            y: {
                beginAtZero: true,
                grid: { color: token('--grid'), drawTicks: false },
                border: { display: false },
                ticks: {
                    color: token('--text-subtle'),
                    font: { size: 11 },
                    padding: 8,
                    callback: v => (v >= 1000 ? `${v / 1000}k` : v),
                },
            },
        },
    };
}

function rebuildChart() {
    const { sets, cutover } = buildDatasets();
    if (chart) chart.destroy();
    chart = new Chart($('forecastChart'), {
        type: 'line',
        data: { datasets: sets },
        options: chartOptions(),
        plugins: [annotate],
    });
    chart.$cutover = cutover;
    chart.update();
    renderSeriesToggles();
}

function restyleChart() {
    if (!chart) return;
    chart.options = chartOptions();
    chart.data.datasets.forEach(d => {
        if (d.$band) {
            if (d.backgroundColor) d.backgroundColor = token('--band');
            d.borderColor = token('--band-edge');
        }
        if (d.$themed) d.borderColor = token('--accent');
    });
    chart.update('none');
    renderSeriesToggles();
}

/* Toggle chips, in place of Chart.js's cramped strikethrough legend.
   Visibility goes through the chart's own accessors: reading meta.hidden
   directly does not work, because it starts as null rather than false — every
   chip would paint as pressed while its series was hidden, and the first click
   would set hidden = true on something already hidden. */
function renderSeriesToggles() {
    const container = $('chartSeries');
    container.innerHTML = chart.data.datasets
        .map((d, i) => ({ d, i }))
        .filter(({ d }) => !d.$band)
        .map(({ d, i }) => `
            <button type="button" data-index="${i}" aria-pressed="${chart.isDatasetVisible(i)}">
                <span class="swatch" style="background:${d.borderColor}"></span>${esc(d.label)}
            </button>`).join('');

    container.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            const i = Number(btn.dataset.index);
            const visible = chart.isDatasetVisible(i);
            chart.setDatasetVisibility(i, !visible);
            btn.setAttribute('aria-pressed', String(!visible));
            chart.update();
        });
    });
}
