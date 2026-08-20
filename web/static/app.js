/**
 * AI-Assisted Retrosynthesis Studio - Client Application Logic
 */

let currentPlans = [];
let activePlanIndex = 0;

// DOM Elements
const smilesInput = document.getElementById("smiles-input");
const planBtn = document.getElementById("plan-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const presetChipsContainer = document.getElementById("preset-chips");

const targetStructureBox = document.getElementById("target-structure-box");
const targetPropsContainer = document.getElementById("target-props-container");
const propFormula = document.getElementById("prop-formula");
const propMw = document.getElementById("prop-mw");
const propLogp = document.getElementById("prop-logp");
const propTpsa = document.getElementById("prop-tpsa");
const propHbaHbd = document.getElementById("prop-hba-hbd");
const propRotb = document.getElementById("prop-rotb");
const targetFgTags = document.getElementById("target-fg-tags");

const routeTabs = document.getElementById("route-tabs");
const routeCountBadge = document.getElementById("route-count-badge");
const routeMetricsBanner = document.getElementById("route-metrics-banner");
const statSteps = document.getElementById("stat-steps");
const statYield = document.getElementById("stat-yield");
const statGreen = document.getElementById("stat-green");
const statAe = document.getElementById("stat-ae");
const statPmi = document.getElementById("stat-pmi");
const routeTreeContainer = document.getElementById("route-tree-container");

const sopSection = document.getElementById("sop-section");
const sopMarkdownViewer = document.getElementById("sop-markdown-viewer");
const copySopBtn = document.getElementById("copy-sop-btn");

const maxDepthSelect = document.getElementById("max-depth-select");
const maxRoutesSelect = document.getElementById("max-routes-select");
const searchTimeoutInput = document.getElementById("search-timeout");

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  loadPresets();
  setupEventListeners();
  // Auto-validate default SMILES
  validateTargetMolecule(smilesInput.value.trim());
});

function setupEventListeners() {
  // Plan button click
  planBtn.addEventListener("click", () => {
    executeRetrosynthesis();
  });

  // SMILES input enter key
  smilesInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      executeRetrosynthesis();
    }
  });

  // Debounced input change for structure preview
  let debounceTimeout;
  smilesInput.addEventListener("input", () => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      validateTargetMolecule(smilesInput.value.trim());
    }, 400);
  });

  // Copy SOP button
  copySopBtn.addEventListener("click", () => {
    if (sopMarkdownViewer.textContent) {
      navigator.clipboard.writeText(sopMarkdownViewer.textContent).then(() => {
        const originalText = copySopBtn.textContent;
        copySopBtn.textContent = "✅ Copied to Clipboard!";
        setTimeout(() => {
          copySopBtn.textContent = originalText;
        }, 2000);
      });
    }
  });
}

// Load Curated Chemical Presets
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();
    if (data.presets && data.presets.length > 0) {
      presetChipsContainer.innerHTML = "";
      data.presets.forEach((preset, idx) => {
        const chip = document.createElement("div");
        chip.className = `preset-chip ${idx === 0 ? "active" : ""}`;
        chip.textContent = preset.name;
        chip.title = `${preset.category}: ${preset.description}`;
        chip.addEventListener("click", () => {
          document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
          chip.classList.add("active");
          smilesInput.value = preset.smiles;
          validateTargetMolecule(preset.smiles);
        });
        presetChipsContainer.appendChild(chip);
      });
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

// Validate Molecule and Render 2D Structure
async function validateTargetMolecule(smiles) {
  if (!smiles) {
    targetStructureBox.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🧪</div>
        <p>Enter SMILES to render 2D structure</p>
      </div>`;
    targetPropsContainer.style.display = "none";
    return;
  }

  try {
    const res = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles }),
    });

    if (!res.ok) {
      targetStructureBox.innerHTML = `
        <div class="empty-state" style="color: var(--accent-rose);">
          <div class="empty-state-icon">⚠️</div>
          <p>Invalid SMILES Structure</p>
        </div>`;
      targetPropsContainer.style.display = "none";
      return;
    }

    const data = await res.json();
    if (data.valid) {
      targetStructureBox.innerHTML = data.svg_light || data.svg_dark;
      propFormula.textContent = data.formula || "-";
      propMw.textContent = data.molecular_weight ? `${data.molecular_weight} g/mol` : "-";
      propLogp.textContent = data.logp !== undefined ? data.logp : "-";
      propTpsa.textContent = data.tpsa ? `${data.tpsa} Å²` : "-";
      propHbaHbd.textContent = `${data.hbd} / ${data.hba}`;
      propRotb.textContent = data.rotatable_bonds !== undefined ? data.rotatable_bonds : "-";

      // Render functional groups
      targetFgTags.innerHTML = "";
      if (data.functional_groups && data.functional_groups.length > 0) {
        data.functional_groups.forEach((fg) => {
          const tag = document.createElement("span");
          tag.className = "fg-tag";
          tag.textContent = fg.replace(/_/g, " ");
          targetFgTags.appendChild(tag);
        });
      } else {
        targetFgTags.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">None detected</span>';
      }

      targetPropsContainer.style.display = "block";
    }
  } catch (err) {
    console.error("Validation error:", err);
  }
}

// Execute Multi-Step Retrosynthesis
async function executeRetrosynthesis() {
  const smiles = smilesInput.value.trim();
  if (!smiles) return;

  setLoading(true);
  routeTreeContainer.innerHTML = `
    <div class="empty-state">
      <div class="spinner" style="width: 36px; height: 36px; margin-bottom: 14px;"></div>
      <h4>Executing Multi-Step Retrosynthesis Search...</h4>
      <p>Applying reaction disconnections down to fundamental chemical feedstocks.</p>
    </div>`;

  try {
    const res = await fetch("/api/plan-synthesis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles: smiles,
        max_depth: parseInt(maxDepthSelect.value, 10),
        max_routes: parseInt(maxRoutesSelect.value, 10),
        time_limit_sec: parseFloat(searchTimeoutInput.value),
      }),
    });

    const data = await res.json();
    setLoading(false);

    if (!res.ok || !data.success) {
      routeTreeContainer.innerHTML = `
        <div class="empty-state" style="color: var(--accent-rose);">
          <div class="empty-state-icon">❌</div>
          <h4>Retrosynthesis Search Incomplete</h4>
          <p>${data.error || "No reaction steps could be formed with current parameters."}</p>
        </div>`;
      routeTabs.style.display = "none";
      routeMetricsBanner.style.display = "none";
      routeCountBadge.style.display = "none";
      sopSection.style.display = "none";
      return;
    }

    currentPlans = data.plans || [];
    renderRouteTabs(currentPlans);

    if (currentPlans.length > 0) {
      selectPlan(0);
    } else {
      routeTreeContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚗️</div>
          <h4>Basic Chemical Feedstock</h4>
          <p>This molecule is already a primary chemical raw material.</p>
        </div>`;
      routeTabs.style.display = "none";
      routeMetricsBanner.style.display = "none";
      routeCountBadge.style.display = "none";
      sopSection.style.display = "none";
    }
  } catch (err) {
    setLoading(false);
    console.error("Retrosynthesis error:", err);
    routeTreeContainer.innerHTML = `
      <div class="empty-state" style="color: var(--accent-rose);">
        <div class="empty-state-icon">⚠️</div>
        <h4>Server Connection Error</h4>
        <p>Could not connect to the retrosynthesis backend.</p>
      </div>`;
  }
}

// Render Route Selector Tabs
function renderRouteTabs(plans) {
  routeTabs.innerHTML = "";
  if (plans.length === 0) {
    routeTabs.style.display = "none";
    routeCountBadge.style.display = "none";
    return;
  }

  routeCountBadge.textContent = `${plans.length} Pathways Solved`;
  routeCountBadge.style.display = "inline-flex";
  routeTabs.style.display = "flex";

  plans.forEach((plan, idx) => {
    const tab = document.createElement("div");
    tab.className = `route-tab ${idx === 0 ? "active" : ""}`;
    
    const paretoTag = plan.metrics.pareto_tags && plan.metrics.pareto_tags[0] 
      ? plan.metrics.pareto_tags[0] 
      : `${plan.metrics.total_steps} Steps`;

    tab.innerHTML = `
      <span>Pathway #${idx + 1} (${plan.metrics.total_steps} Steps)</span>
      <span class="route-tag-pill">${paretoTag}</span>
    `;

    tab.addEventListener("click", () => {
      document.querySelectorAll(".route-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      selectPlan(idx);
    });

    routeTabs.appendChild(tab);
  });
}

// Select and Display a Specific Route Plan
function selectPlan(index) {
  activePlanIndex = index;
  const plan = currentPlans[index];
  if (!plan) return;

  // 1. Update Metrics Banner
  routeMetricsBanner.style.display = "grid";
  statSteps.textContent = plan.metrics.total_steps;
  statYield.textContent = `${plan.metrics.cumulative_yield}%`;
  statGreen.textContent = `${plan.metrics.green_chemistry_score}/100`;
  statAe.textContent = `${plan.metrics.average_atom_economy}%`;
  statPmi.textContent = plan.metrics.estimated_pmi;

  // 2. Render Interactive Reaction Tree
  routeTreeContainer.innerHTML = "";

  if (plan.steps.length === 0) {
    routeTreeContainer.innerHTML = `
      <div class="step-card">
        <div class="step-header">
          <span class="step-number">Primary Feedstock Chemical</span>
        </div>
        <p style="color: var(--accent-green);">Primary raw material.</p>
      </div>`;
    sopSection.style.display = "none";
    return;
  }

  plan.steps.forEach((step) => {
    const stepCard = document.createElement("div");
    stepCard.className = "step-card";

    // Build Reactants Boxes HTML
    let reactantsHtml = step.reactants_smiles
      .map((rSmi, rIdx) => {
        const svg = step.reactants_svgs && step.reactants_svgs[rIdx] ? step.reactants_svgs[rIdx] : "";
        return `
          <div class="mol-box">
            ${svg}
            <div class="mol-smiles-label">${rSmi}</div>
          </div>
        `;
      })
      .join('<div style="font-size: 18px; font-weight: bold; color: var(--text-muted);">+</div>');

    // Product Box HTML
    const prodSvg = step.product_svg || "";
    const productHtml = `
      <div class="mol-box" style="border-color: #0284c7;">
        ${prodSvg}
        <div class="mol-smiles-label" style="color: #0284c7; font-weight: 700;">${step.product_smiles}</div>
      </div>
    `;

    // Moisture & Atmosphere Badge
    let moistClass = "tag-water-tolerant";
    let moistIcon = "💧";
    if (step.moisture_category.includes("Strictly Anhydrous")) {
      moistClass = "tag-anhydrous";
      moistIcon = "🔥";
    } else if (step.moisture_category.includes("Moisture-sensitive")) {
      moistClass = "tag-temp";
      moistIcon = "🧪";
    }

    // Protecting Group Alert
    let pgHtml = "";
    if (step.protection_plan) {
      const pg = step.protection_plan;
      pgHtml = `
        <div class="alert-box alert-protect">
          <span>🛡️</span>
          <div>
            <strong>Protecting Group Required (${pg.protecting_group}):</strong> 
            Install with <em>${pg.installation.reagents}</em> in ${pg.installation.solvent} (${pg.installation.yield}); 
            deprotect with <em>${pg.deprotection.reagents}</em> (${pg.deprotection.yield}).
          </div>
        </div>
      `;
    }

    // Cascade Alert
    let cascadeHtml = "";
    if (step.is_cascade && step.cascade_note) {
      cascadeHtml = `
        <div class="alert-box alert-cascade">
          <span>⚡</span>
          <div><strong>One-Pot Telescoping:</strong> ${step.cascade_note}</div>
        </div>
      `;
    }

    stepCard.innerHTML = `
      <div class="step-header">
        <div class="step-number">
          <span>Step ${step.step_number}</span>
          <span style="color: var(--text-primary); font-weight: 600;">— ${step.reaction_name}</span>
        </div>
        <span class="step-class-tag">${step.reaction_class}</span>
      </div>

      <div class="reaction-flow">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: center;">
          ${reactantsHtml}
        </div>
        <div class="rxn-arrow">
          <span>➔</span>
          <span class="yield-label">${Math.round(step.step_yield * 100)}% Yield</span>
        </div>
        <div>
          ${productHtml}
        </div>
      </div>

      <div class="step-tags">
        <span class="condition-tag tag-reagent">🧪 Reagents: ${step.reagents.join(", ")}</span>
        <span class="condition-tag tag-solvent">💧 Solvents: ${step.solvents.join(", ")}</span>
        <span class="condition-tag tag-temp">🌡️ Temp: ${step.temperature}</span>
        <span class="condition-tag ${moistClass}">${moistIcon} ${step.moisture_category}</span>
        <span class="condition-tag" style="background: #f1f5f9; color: #475569;">⚛️ Atom Econ: ${step.atom_economy}%</span>
      </div>

      <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">
        <strong>Workup & Isolation:</strong> ${step.workup_protocol}
      </div>

      ${pgHtml}
      ${cascadeHtml}
    `;

    routeTreeContainer.appendChild(stepCard);
  });

  // 3. Fetch and Render SOP
  fetchSop(plan);
}

// Fetch Laboratory SOP Markdown
async function fetchSop(plan) {
  sopSection.style.display = "block";
  sopMarkdownViewer.textContent = "Generating standard operating procedure...";

  try {
    const res = await fetch("/api/export-sop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan }),
    });
    const data = await res.json();
    if (data.sop_markdown) {
      sopMarkdownViewer.textContent = data.sop_markdown;
    }
  } catch (err) {
    sopMarkdownViewer.textContent = "Failed to export laboratory protocol.";
  }
}

// Loading State Helper
function setLoading(isLoading) {
  planBtn.disabled = isLoading;
  btnSpinner.style.display = isLoading ? "inline-block" : "none";
  btnText.textContent = isLoading ? "Computing Routes..." : "🚀 Plan Multi-Step Synthesis";
}
