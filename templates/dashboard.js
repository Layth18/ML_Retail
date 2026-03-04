let dashData = null;
let activeTab = "seasons";

/* ── DATA PROCESSING ── */
/**
 * Safely handles numbers from the API. 
 * Since Flask now sends real currency values, we just ensure they are valid.
 */
function cleanVal(val) {
    const num = parseFloat(val);
    return (!isNaN(num) && isFinite(num)) ? num : 0;
}

/* ── TAB SWITCHER ── */
function switchTab(tab, btn) {
    activeTab = tab;
    
    // UI Feedback for buttons
    document.querySelectorAll(".dash-tab").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");

    // Toggle visibility
    const seasonsEl = document.getElementById("dash-seasons");
    const regionsEl = document.getElementById("dash-regions");

    if (seasonsEl && regionsEl) {
        seasonsEl.classList.toggle("hidden", tab !== "seasons");
        regionsEl.classList.toggle("hidden", tab !== "regions");
    }
}

/* ── FORMAT GAIN ── */
function fmtGain(val) {
    const value = cleanVal(val);
    if (value === 0) return "$0.00";
    
    // For extreme outliers/huge numbers
    if (value > 10000000) {
        return "$" + value.toExponential(2);
    }
    
    // Standard currency display
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
    }).format(value);
}

/* ── SEASONS VIEW ── */
const seasonMeta = {
    Spring: { icon: "🌸", cls: "season-spring" },
    Summer: { icon: "☀️", cls: "season-summer" },
    Autumn: { icon: "🍂", cls: "season-autumn" },
    Winter: { icon: "❄️", cls: "season-winter" }
};

const rankCls = ["gold", "silver", "bronze"];

function renderSeasons(seasons) {
    const grid = document.getElementById("seasons-grid");
    const annualEl = document.getElementById("annual-stat");
    if (!grid || !seasons) return;

    const seasonOrder = ["Spring", "Summer", "Autumn", "Winter"];

    // Max gain for relative bar widths
    const allGains = seasonOrder.flatMap(s =>
        (seasons[s]?.top_regions || []).map(r => cleanVal(r.estimated_gain))
    );
    const maxGain = Math.max(...allGains, 1);

    let grandAnnual = 0;

    grid.innerHTML = seasonOrder.map(sName => {
        const sd = seasons[sName];
        if (!sd) return "";
        
        // Summing the projection provided by API
        grandAnnual += cleanVal(sd.season_total);
        const sm = seasonMeta[sName] || { icon: "🌍", cls: "" };

        const rows = (sd.top_regions || []).map((r, i) => {
            const gain = cleanVal(r.estimated_gain);
            const pct = Math.min(Math.round((gain / maxGain) * 100), 100);
            const rc = rankCls[i] || "";

            return `
                <div class="season-region-row">
                    <span class="region-rank ${rc}">${i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}</span>
                    <span class="region-name-dash">${r.region}</span>
                    <div class="region-bar-wrap">
                        <div class="region-bar" style="width:${pct}%"></div>
                    </div>
                    <span class="region-gain">${fmtGain(gain)}</span>
                </div>`;
        }).join("");

        return `
            <div class="season-card ${sm.cls}">
                <div class="season-card-header">
                    <div class="season-name">${sm.icon} ${sName}</div>
                    <div class="season-totals">
                        <div class="season-total-label">Top-3 Total</div>
                        <div class="season-total-val">${fmtGain(sd.season_total)}</div>
                    </div>
                </div>
                <div class="season-regions">${rows}</div>
            </div>`;
    }).join("");

    if (annualEl) {
        annualEl.innerHTML = `
            <div class="annual-stat-banner">
                <div class="annual-stat-left">
                    <div class="annual-stat-icon">📈</div>
                    <div>
                        <div class="annual-stat-label">Projected Annual Revenue Gain</div>
                        <div class="annual-stat-sub">Sum of all seasonal top-region projections</div>
                    </div>
                </div>
                <div class="annual-stat-value">${fmtGain(grandAnnual)}</div>
            </div>`;
    }
}

/* ── REGIONS VIEW ── */
const seasonDotCls = { Spring: "dot-spring", Summer: "dot-summer", Autumn: "dot-autumn", Winter: "dot-winter" };

function renderRegions(regions) {
    const list = document.getElementById("regions-list");
    if (!list || !regions) return;

    list.innerHTML = `<div class="regions-grid">${regions.map((r, i) => {
        const totalGain = cleanVal(r.total_annual_gain);

        const pills = (r.seasons_ranked || []).map((s, si) => {
            const gain = cleanVal(s.estimated_gain);
            const dotCls = seasonDotCls[s.season] || "";

            return `
                <div class="region-season-pill">
                    <span class="season-dot ${dotCls}"></span>
                    <span class="season-pill-name">${s.season}</span>
                    <span class="season-pill-gain">${fmtGain(gain)}</span>
                    <span class="season-pill-rank rank-${si + 1}">#${si + 1}</span>
                </div>`;
        }).join("");

        return `
            <div class="region-card">
                <div class="region-card-header">
                    <div class="region-card-top">
                        <span class="region-rank-badge">#${i + 1}</span>
                    </div>
                    <div class="region-card-name">${r.region}</div>
                    <div class="region-total-row">
                        <span class="region-total-gain">${fmtGain(totalGain)}</span>
                        <span class="region-total-label">/ yr</span>
                    </div>
                </div>
                <div class="region-seasons">${pills}</div>
            </div>`;
    }).join("")}</div>`;
}

/* ── LOAD DASHBOARD ── */
async function loadDashboard() {
    const loader = document.getElementById("dash-loading");
    const seasonsContainer = document.getElementById("dash-seasons");
    const regionsContainer = document.getElementById("dash-regions");

    if (loader) loader.style.display = "flex";
    
    // Ensure containers start hidden
    if (seasonsContainer) seasonsContainer.classList.add("hidden");
    if (regionsContainer) regionsContainer.classList.add("hidden");

    try {
        const response = await fetch("http://127.0.0.1:5000/api/marketing_dashboard");
        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        dashData = await response.json();

        renderSeasons(dashData.seasons);
        renderRegions(dashData.regions);

        if (loader) loader.style.display = "none";
        
        // Use the Tab switcher to show the correct first view
        switchTab(activeTab, document.querySelector(`.dash-tab[onclick*="${activeTab}"]`));

    } catch (err) {
        console.error("Dashboard Load Error:", err);
        if (loader) {
            loader.innerHTML = `
                <div class="error-msg" style="text-align:center; padding: 2rem;">
                    <p style="font-size:16px; font-weight:600; color:#b91c1c;">⚠️ Connection Failed</p>
                    <p style="color:#6b7280; font-size:13px;">The prediction engine at 127.0.0.1:5000 is unreachable.</p>
                </div>`;
        }
    }
}

document.addEventListener("DOMContentLoaded", loadDashboard);