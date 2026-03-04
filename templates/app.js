/* ── CHURN META ── */
const churnMeta = {
  0: { cls: "risk-critical", badge: "🔴 Critique", riskLabel: "Critical Risk" },
  1: { cls: "risk-low", badge: "🟢 Faible", riskLabel: "Low Risk" },
  2: { cls: "risk-medium", badge: "🟡 Moyen", riskLabel: "Medium Risk" },
  3: { cls: "risk-high", badge: "🟠 Élevé", riskLabel: "High Risk" },
};

/* ── PERSONA META ── */
const personaMeta = {
  0: { cls: "persona-0", icon: "💰", label: "Loyal High-Spender" },
  1: { cls: "persona-1", icon: "🧭", label: "Recent Explorer / Newbie" },
  2: { cls: "persona-2", icon: "🧭", label: "At-Risk / Hibernating" },
  3: { cls: "persona-3", icon: "💎", label: "Active Loyal" },
};

/* ── 16-COMBO STRATEGY MATRIX ── (persona_id, churn_id) */
const strategyMatrix = {
  "3,0": {
    title: "VIP Recovery",
    icon: "🚨",
    strategy:
      "URGENT: VIP Recovery. Personal phone call and high-value retention offer.",
  },
  "3,1": {
    title: "VIP Maintenance",
    icon: "👑",
    strategy:
      "VIP Maintenance: Early access to new collections and loyalty points.",
  },
  "3,2": {
    title: "Proactive Care",
    icon: "🎁",
    strategy:
      "Proactive Care: Survey to check satisfaction and a 'Thank You' gift.",
  },
  "3,3": {
    title: "Retention",
    icon: "⚠️",
    strategy: "Retention: Exclusive discount to prevent further drift.",
  },

  "1,0": {
    title: "Last Chance",
    icon: "🆘",
    strategy: "Last Chance: Strong 'Welcome Back' discount before they churn.",
  },
  "1,1": {
    title: "Onboarding",
    icon: "🌱",
    strategy:
      "Onboarding: Educational content and second-purchase encouragement.",
  },
  "1,2": {
    title: "Nurture",
    icon: "🤝",
    strategy: "Nurture: Social proof and reviews to build trust.",
  },
  "1,3": {
    title: "Urgency",
    icon: "⏳",
    strategy:
      "Urgency: Limited-time offer to convert them into a repeat buyer.",
  },

  "2,0": {
    title: "Final Attempt",
    icon: "🏳️",
    strategy: "Final Attempt: Extreme clearance offer or 'Goodbye' survey.",
  },
  "2,1": {
    title: "Re-engagement",
    icon: "📬",
    strategy:
      "Re-engagement: Highlight new arrivals in their favorite category.",
  },
  "2,2": {
    title: "Win-Back",
    icon: "💌",
    strategy:
      "Win-Back: Standard 'We Miss You' email sequence with free shipping.",
  },
  "2,3": {
    title: "Aggressive Win-Back",
    icon: "💥",
    strategy: "Aggressive Win-Back: Deep discounts (50%+) to trigger a return.",
  },

  "0,0": {
    title: "Premium VIP Care",
    icon: "🌟",
    strategy: "Premium VIP Care: Personal manager and exclusive gifts.",
  },
  "0,1": {
    title: "Upsell Opportunity",
    icon: "💹",
    strategy:
      "Upsell Opportunity: Suggest premium products and loyalty bonuses.",
  },
  "0,2": {
    title: "Retention Boost",
    icon: "🎯",
    strategy:
      "Retention Boost: Personalized discount based on previous high spend.",
  },
  "0,3": {
    title: "High-Value Maintenance",
    icon: "🏆",
    strategy:
      "High-Value Maintenance: Maintain engagement with early offers and VIP treatment.",
  },
};

/* ── UNSKEW FUNCTION ── */
function unskewMonetary(val) {
  if (!val || isNaN(val)) return 0;
  return Math.max(Math.exp(val) - 1, 0);
}

/* ── FORM FIELDS ── */
const categoryMappings = {
  FavoriteSeason: { 0: "Autumn", 1: "Winter", 2: "Spring", 3: "Summer" },
  Region: { 4: "Central Europe", 8: "UK", Other: "Other" },
};

const features = ["Recency", "Frequency", "CustomerTenureDays"];

const container = document.getElementById("fields");
container.innerHTML = "";

features.forEach((f) => {
  const label = f.replace(/([A-Z])/g, " $1").trim();
  let fieldHTML = "";

  if (categoryMappings[f]) {
    const options = Object.entries(categoryMappings[f])
      .map(([v, l]) => `<option value="${v}">${l}</option>`)
      .join("");
    fieldHTML = `<div class="field-wrap"><label>${label}</label><select name="${f}">${options}</select></div>`;
  } else if (f === "WeekendPurchaseRatio") {
    fieldHTML = `<div class="field-wrap"><label>${label}</label><input type="number" step="0.01" min="0" max="1" value="0" name="${f}"></div>`;
  } else {
    fieldHTML = `<div class="field-wrap"><label>${label}</label><input type="number" step="any" value="0" name="${f}"></div>`;
  }

  container.innerHTML += fieldHTML;
});

/* ── SUBMIT HANDLER ── */
document.getElementById("analysis-form").onsubmit = async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector("button");
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "Analyzing customer profile…";

  const fd = new FormData(e.target);
  const vals = features.map((f) => parseFloat(fd.get(f) || 0));

  try {
    const res = await fetch("http://127.0.0.1:5000/api/predict_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: vals }),
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const churnId = data.raw_indices.churn;
    const personaId = data.raw_indices.persona;

    const cm = churnMeta[churnId] ?? churnMeta[1];
    const pm = personaMeta[personaId] ?? personaMeta[0];
    const key = `${personaId},${churnId}`;
    const combo = strategyMatrix[key];

    document.getElementById("churn-val").innerText = data.churn_risk;
    document.getElementById("persona-val").innerText = data.persona_name;
    document.getElementById("churn-badge").innerText = cm.badge;

    document.getElementById("strategy-val").innerText = combo
      ? combo.strategy
      : "--";

    const area = document.getElementById("results-area");
    const allCombos = [];
    for (let p = 0; p < 4; p++)
      for (let c = 0; c < 4; c++) allCombos.push(`combo-${p}-${c}`);
    area.classList.remove(...allCombos);
    area.classList.add(`combo-${personaId}-${churnId}`);

    if (combo) {
      document.getElementById("combo-icon").innerText = combo.icon;
      document.getElementById("combo-title").innerText = combo.title;
      document.getElementById("combo-strategy-text").innerText = combo.strategy;
      document.getElementById("combo-badge-el").innerText =
        `${pm.label} · ${cm.riskLabel}`;
      const banner = document.getElementById("combo-banner");
      banner.classList.remove("visible");
      banner.offsetHeight;
      banner.classList.add("visible");
    }

    document.querySelectorAll(".result-card").forEach((c) => {
      c.style.animation = "none";
      c.offsetHeight;
      c.style.animation = "";
    });

    area.classList.remove("hidden");
    area.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
};
