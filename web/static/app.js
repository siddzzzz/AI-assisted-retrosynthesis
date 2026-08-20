/**
 * AI-Assisted Retrosynthesis Studio - Client Application Logic
 */

let currentPlans = [];
let activePlanIndex = 0;
let currentTargetScale = 1.0;

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
const scaleController = document.getElementById("scale-controller");
const customScaleInput = document.getElementById("custom-scale-input");
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
  validateTargetMolecule(smilesInput.value.trim());
});

function setupEventListeners() {
  planBtn.addEventListener("click", () => {
    executeRetrosynthesis();
  });

  smilesInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      executeRetrosynthesis();
    }
  });

  let debounceTimeout;
  smilesInput.addEventListener("input", () => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      validateTargetMolecule(smilesInput.value.trim());
    }, 400);
  });

  // Scale Controller chips
  document.querySelectorAll(".scale-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".scale-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const scale = parseFloat(chip.dataset.scale);
      currentTargetScale = scale;
      customScaleInput.value = scale;
      rescaleCurrentPlan(scale);
    });
  });

  customScaleInput.addEventListener("change", () => {
    const scale = parseFloat(customScaleInput.value) || 1.0;
    currentTargetScale = scale;
    document.querySelectorAll(".scale-chip").forEach((c) => c.classList.remove("active"));
    rescaleCurrentPlan(scale);
  });

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

// Rescale Current Plan Stoichiometry
async function rescaleCurrentPlan(scaleGrams) {
  if (!currentPlans[activePlanIndex]) return;
  
  try {
    const res = await fetch("/api/scale-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan: currentPlans[activePlanIndex],
        target_scale_g: scaleGrams,
      }),
    });
    const data = await res.json();
    if (data.plan) {
      currentPlans[activePlanIndex] = data.plan;
      selectPlan(activePlanIndex);
    }
  } catch (err) {
    console.error("Failed to rescale batch:", err);
  }
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
      <p>Applying reaction disconnections, checking 3D sterics, and matching patent precedents.</p>
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
      scaleController.style.display = "none";
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
          <h4>Primary Commodity Feedstock</h4>
          <p>This molecule is already a basic raw chemical material.</p>
        </div>`;
      routeTabs.style.display = "none";
      routeMetricsBanner.style.display = "none";
      routeCountBadge.style.display = "none";
      scaleController.style.display = "none";
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
    scaleController.style.display = "none";
    return;
  }

  routeCountBadge.textContent = `${plans.length} Pathways Solved`;
  routeCountBadge.style.display = "inline-flex";
  routeTabs.style.display = "flex";
  scaleController.style.display = "flex";

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

  // 2. Render Interactive Reaction Tree with Stoichiometry & Precedent
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

    // Precedent & 3D Sterics Badges
    const precedent = step.precedent || {};
    const sterics = step.sterics || {};
    const purif = step.purification || {};
    const stoich = step.stoichiometry || {};

    let precedentTag = "";
    if (precedent.patent_id) {
      precedentTag = `<span class="condition-tag tag-precedent">📚 Patent: ${precedent.patent_id} (${precedent.similarity_pct || 85}%)</span>`;
    }

    let stericTag = "";
    if (sterics.accessibility) {
      stericTag = `<span class="condition-tag tag-steric">🧊 3D Sterics: ${sterics.accessibility} (${sterics.steric_burial_score}%)</span>`;
    }

    // Stoichiometry Table HTML
    let stoichRows = (stoich.stoichiometry_table || []).map((row, rIdx) => {
      const massDisp = row.mass_g >= 1.0 ? `${row.mass_g} g` : `${row.mass_mg} mg`;
      const volDisp = row.volume_ml ? `${row.volume_ml} mL` : "—";
      const nameDisp = row.name || row.smiles;
      const isLimiting = row.role.includes("Limiting");
      return `
        <tr class="${isLimiting ? 'limiting-row' : ''}">
          <td style="font-weight: 600;">${nameDisp}</td>
          <td>${row.role}</td>
          <td>${row.mw}</td>
          <td>${row.equivalents.toFixed(2)}</td>
          <td>${row.mmol.toFixed(2)}</td>
          <td>${massDisp}</td>
          <td>${volDisp}</td>
        </tr>
      `;
    }).join("");

    let stoichHtml = `
      <div class="stoich-container">
        <table class="stoich-table">
          <thead>
            <tr>
              <th>Component</th>
              <th>Role</th>
              <th>MW (g/mol)</th>
              <th>Eq.</th>
              <th>mmol</th>
              <th>Mass</th>
              <th>Volume</th>
            </tr>
          </thead>
          <tbody>
            ${stoichRows}
          </tbody>
        </table>
      </div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
        💡 Reaction Volume: <strong>${stoich.recommended_solvent_volume_ml || 20.0} mL</strong> ${stoich.primary_solvent || 'solvent'} (~${stoich.concentration_molar || 0.2} M concentration).
      </div>
    `;

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

    // Purification / Chromatography Advice
    let purifHtml = "";
    if (purif.recommended_eluent) {
      purifHtml = `
        <div class="alert-box alert-purify">
          <span>🧪</span>
          <div>
            <strong>Flash Chromatography Eluent:</strong> <code>${purif.recommended_eluent}</code> | 
            <strong>TLC Stain:</strong> ${purif.tlc_visualization} | 
            <strong>Byproducts:</strong> ${purif.expected_side_products ? purif.expected_side_products.join(", ") : "Standard side-products"}
          </div>
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
        ${precedentTag}
        ${stericTag}
      </div>

      ${stoichHtml}
      ${purifHtml}
      ${pgHtml}
      ${cascadeHtml}
    `;

    routeTreeContainer.appendChild(stepCard);
  });

  // 3. Fetch and Render ELN SOP
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
      body: JSON.stringify({ plan: plan, target_scale_g: currentTargetScale }),
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
