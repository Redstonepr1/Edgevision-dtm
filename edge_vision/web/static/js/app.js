/**
 * Edge-Vision Ambient Web Application Logic
 * Zero-Touch AI Food Freshness & Waste Analytics Platform
 */

let currentFilter = 'ALL';
let inventoryData = [];
let decayChartInstance = null;
let weeklyChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initLucide();
    loadDashboardData();
    setupEventListeners();

    // Auto-refresh inventory every 20 seconds
    setInterval(() => {
        loadInventory(false);
    }, 20000);
});

function initLucide() {
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

async function loadDashboardData() {
    await Promise.all([
        loadInventory(true),
        loadAnalytics(),
        loadDecayTrajectories(),
        loadExpiringRecipes()
    ]);
}

function setupEventListeners() {
    // Tab switching
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active-tab', 'text-sky-400', 'border-sky-400'));
            tabs.forEach(t => t.classList.add('text-slate-400', 'border-transparent'));
            tab.classList.add('active-tab', 'text-sky-400', 'border-sky-400');
            tab.classList.remove('text-slate-400', 'border-transparent');

            const target = tab.getAttribute('data-target');
            document.querySelectorAll('.view-section').forEach(sec => sec.classList.add('hidden'));
            const targetEl = document.getElementById(target);
            if (targetEl) targetEl.classList.remove('hidden');

            if (target === 'view-decay') {
                setTimeout(initDecayChart, 100);
            } else if (target === 'view-analytics') {
                setTimeout(initWeeklyChart, 100);
            }
        });
    });

    // Filter buttons
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('bg-sky-500/20', 'text-sky-400', 'border-sky-500/40'));
            filterBtns.forEach(b => b.classList.add('text-slate-400', 'border-slate-800'));
            btn.classList.add('bg-sky-500/20', 'text-sky-400', 'border-sky-500/40');
            btn.classList.remove('text-slate-400', 'border-slate-800');
            currentFilter = btn.getAttribute('data-filter');
            renderInventoryGrid();
        });
    });

    // Image Upload Form (Simulated ESP32 capture)
    const uploadInput = document.getElementById('frame-upload-input');
    if (uploadInput) {
        uploadInput.addEventListener('change', handleFrameUpload);
    }
}

// ==============================================================================
// 1. INVENTORY MANAGEMENT & RENDERING
// ==============================================================================

async function loadInventory(showLoading = true) {
    try {
        const res = await fetch('/api/v1/inventory');
        if (!res.ok) throw new Error('Failed to load inventory');
        inventoryData = await res.json();
        renderInventoryGrid();
        updateInventoryStats();
    } catch (err) {
        console.error('Inventory load error:', err);
    }
}

function renderInventoryGrid() {
    const grid = document.getElementById('inventory-grid');
    if (!grid) return;

    let filtered = inventoryData;
    if (currentFilter !== 'ALL') {
        filtered = inventoryData.filter(i => i.status === currentFilter);
    }

    if (filtered.length === 0) {
        grid.innerHTML = `
            <div class="col-span-full py-16 text-center text-slate-500 glass-card rounded-2xl p-8">
                <i data-lucide="package-open" class="w-12 h-12 mx-auto mb-3 opacity-40"></i>
                <p class="text-base font-medium text-slate-400">No items match the current filter (${currentFilter})</p>
                <p class="text-xs text-slate-500 mt-1">Upload a shelf frame to ingest fresh produce.</p>
            </div>
        `;
        initLucide();
        return;
    }

    grid.innerHTML = filtered.map(item => {
        let badgeClass = 'badge-fresh';
        let statusBadge = `${item.category.toUpperCase()} • ${Math.round(item.freshness_percentage)}% Fresh`;
        let daysColor = 'text-emerald-400';

        if (item.status === 'CRITICAL' || item.freshness_percentage < 30) {
            badgeClass = 'badge-critical';
            statusBadge = `${item.category.toUpperCase()} ${Math.round(item.confidence_score)}% • ${Math.round(item.freshness_percentage)}% Fresh ⚠`;
            daysColor = 'text-rose-400 font-bold';
        } else if (item.status === 'APPROACHING_SPOILAGE' || item.freshness_percentage < 70) {
            badgeClass = 'badge-warning';
            statusBadge = `${item.category.toUpperCase()} • ${Math.round(item.freshness_percentage)}% Fresh`;
            daysColor = 'text-amber-400';
        }

        const ocrBadge = item.ocr_expiry_date
            ? `<span class="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/30">
                 <i data-lucide="scan-text" class="w-3 h-3"></i> OCR: ${item.ocr_expiry_date}
               </span>`
            : '';

        return `
            <div class="glass-card rounded-2xl overflow-hidden flex flex-col justify-between border border-slate-800/80 group">
                <!-- Top Image & Detection Badge Overlay -->
                <div class="relative h-44 w-full bg-slate-950 overflow-hidden">
                    <img src="${item.image_url}" alt="${item.category}" 
                         class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                         onerror="this.src='/static/uploads/fresh_apple.jpg'">
                    <div class="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-black/30"></div>
                    
                    <!-- Overlay Badge matching detection output -->
                    <div class="absolute top-3 left-3">
                        <span class="px-2.5 py-1 rounded-full text-xs font-semibold font-mono tracking-wide backdrop-blur-md shadow-lg ${badgeClass}">
                            ${statusBadge}
                        </span>
                    </div>

                    <!-- YOLO Confidence Pill -->
                    <div class="absolute top-3 right-3">
                        <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-black/60 backdrop-blur-md text-slate-300 border border-white/10">
                            YOLOv8 ${item.confidence_score.toFixed(1)}%
                        </span>
                    </div>

                    <!-- Freshness Progress Indicator -->
                    <div class="absolute bottom-0 inset-x-0 h-1 bg-slate-800">
                        <div class="h-full ${item.freshness_percentage < 30 ? 'bg-rose-500' : item.freshness_percentage < 70 ? 'bg-amber-500' : 'bg-emerald-500'}" 
                             style="width: ${item.freshness_percentage}%"></div>
                    </div>
                </div>

                <!-- Card Body -->
                <div class="p-5 flex-1 flex flex-col justify-between">
                    <div>
                        <div class="flex items-start justify-between gap-2 mb-2">
                            <h3 class="text-lg font-bold text-white tracking-tight">${item.category}</h3>
                            <span class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800/60 text-slate-400">
                                ${item.device_id || 'ESP32-CAM'}
                            </span>
                        </div>

                        <!-- Date & Shelf Life Predictions -->
                        <div class="space-y-2 mt-3 pt-3 border-t border-slate-800/60">
                            <div class="flex items-center justify-between text-xs">
                                <span class="text-slate-400 flex items-center gap-1.5">
                                    <i data-lucide="clock" class="w-3.5 h-3.5 text-slate-500"></i> AI Est. Remaining:
                                </span>
                                <span class="${daysColor} font-mono text-sm">
                                    ${item.estimated_days_left} day${item.estimated_days_left !== 1 ? 's' : ''} left
                                </span>
                            </div>
                            ${ocrBadge ? `<div class="flex items-center justify-between pt-1">${ocrBadge}</div>` : ''}
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="mt-5 pt-3 border-t border-slate-800/80 flex items-center justify-between gap-2">
                        <button onclick="openRecipeDrawerFor('${item.category}')" 
                                class="flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 border border-emerald-500/30 transition-colors flex items-center justify-center gap-1.5">
                            <i data-lucide="chef-hat" class="w-3.5 h-3.5"></i> Recipes
                        </button>
                        <button onclick="deleteItem('${item.item_id}')" title="Mark as Cooked / Rescued"
                                class="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors">
                            <i data-lucide="check-check" class="w-4 h-4"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    initLucide();
}

function updateInventoryStats() {
    const total = inventoryData.length;
    const critical = inventoryData.filter(i => i.status === 'CRITICAL').length;
    const approaching = inventoryData.filter(i => i.status === 'APPROACHING_SPOILAGE').length;
    const fresh = inventoryData.filter(i => i.status === 'FRESH').length;

    const elTotal = document.getElementById('stat-total-items');
    const elCrit = document.getElementById('stat-critical-items');
    const elAppr = document.getElementById('stat-approaching-items');
    const elFresh = document.getElementById('stat-fresh-items');

    if (elTotal) elTotal.innerText = total;
    if (elCrit) elCrit.innerText = critical;
    if (elAppr) elAppr.innerText = approaching;
    if (elFresh) elFresh.innerText = fresh;

    // Update Proactive Alert Banner if critical items exist
    const banner = document.getElementById('critical-warning-banner');
    if (banner) {
        if (critical > 0) {
            const critNames = inventoryData.filter(i => i.status === 'CRITICAL').map(i => i.category).join(', ');
            document.getElementById('banner-message').innerHTML = `
                <strong>⚠ Proactive Waste Alert:</strong> ${critical} item(s) (<strong>${critNames}</strong>) spoiling in &le; 2 days! Zero-waste recipes ready.
            `;
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }
    }
}

async function deleteItem(itemId) {
    try {
        const res = await fetch(`/api/v1/inventory/${itemId}`, { method: 'DELETE' });
        if (res.ok) {
            inventoryData = inventoryData.filter(i => i.item_id !== itemId);
            renderInventoryGrid();
            updateInventoryStats();
            loadAnalytics();
            loadExpiringRecipes();
        }
    } catch (e) {
        console.error('Delete error:', e);
    }
}

async function resetSampleInventory() {
    try {
        await fetch('/api/v1/inventory/reset-sample', { method: 'POST' });
        await loadDashboardData();
    } catch (e) {
        console.error('Reset error:', e);
    }
}

async function cleanDuplicates() {
    try {
        const res = await fetch('/api/v1/inventory/deduplicate', { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            console.log(data.message);
            await loadDashboardData();
        }
    } catch (e) {
        console.error('Deduplicate error:', e);
    }
}

// ==============================================================================
// 2. ZERO-WASTE RECIPE DRAWER / MODAL
// ==============================================================================

let currentRecipes = [];

async function loadExpiringRecipes() {
    try {
        const res = await fetch('/api/v1/recipes/expiring');
        if (!res.ok) return;
        const data = await res.json();
        currentRecipes = data.recipes || [];
        renderRecipeDrawer();
    } catch (e) {
        console.error('Recipe load error:', e);
    }
}

function renderRecipeDrawer() {
    const container = document.getElementById('recipe-cards-container');
    if (!container) return;

    if (currentRecipes.length === 0) {
        container.innerHTML = `
            <div class="text-center py-12 text-slate-500">
                <i data-lucide="utensils" class="w-10 h-10 mx-auto mb-2 opacity-40"></i>
                <p class="font-medium text-sm">No items currently requiring immediate zero-waste rescue.</p>
                <p class="text-xs text-slate-600 mt-1">Recipes will generate automatically when items enter critical decay.</p>
            </div>
        `;
        initLucide();
        return;
    }

    container.innerHTML = currentRecipes.map((rec, idx) => `
        <div class="glass-card rounded-xl p-5 border border-slate-800/80 mb-4">
            <div class="flex items-start justify-between gap-3 mb-2">
                <h4 class="text-base font-bold text-white flex items-center gap-2">
                    <span class="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 text-xs flex items-center justify-center font-mono">
                        ${idx + 1}
                    </span>
                    ${rec.title}
                </h4>
                <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 shrink-0">
                    -${rec.waste_prevented_kg} kg waste
                </span>
            </div>
            
            <p class="text-xs text-slate-400 mb-3 leading-relaxed">${rec.description}</p>
            
            <div class="flex items-center gap-4 text-xs font-mono text-slate-400 mb-3">
                <span class="flex items-center gap-1"><i data-lucide="timer" class="w-3.5 h-3.5 text-slate-500"></i> Prep: ${rec.prep_time_minutes}m</span>
                <span class="flex items-center gap-1"><i data-lucide="flame" class="w-3.5 h-3.5 text-amber-500"></i> Cook: ${rec.cook_time_minutes}m</span>
            </div>

            <!-- Ingredients Checklist -->
            <div class="bg-slate-950/60 rounded-lg p-3 mb-3 border border-slate-900">
                <p class="text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1">
                    <i data-lucide="check-square" class="w-3 h-3 text-emerald-400"></i> Ingredients Checklist:
                </p>
                <div class="space-y-1.5">
                    ${rec.expiring_ingredients_used.map(ing => `
                        <label class="flex items-center gap-2 text-xs text-rose-300 cursor-pointer">
                            <input type="checkbox" class="rounded bg-slate-900 border-rose-700 text-rose-500 focus:ring-0">
                            <span class="line-through-checked font-medium">${ing} (Expiring)</span>
                        </label>
                    `).join('')}
                    ${rec.pantry_ingredients_needed.map(ing => `
                        <label class="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                            <input type="checkbox" class="rounded bg-slate-900 border-slate-700 text-emerald-500 focus:ring-0">
                            <span>${ing}</span>
                        </label>
                    `).join('')}
                </div>
            </div>

            <!-- Cooking Instructions -->
            <div>
                <p class="text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                    <i data-lucide="list-ordered" class="w-3 h-3 text-sky-400"></i> Step-by-Step Instructions:
                </p>
                <ol class="space-y-1 text-xs text-slate-400 pl-4 list-decimal marker:text-slate-600">
                    ${rec.instructions.map(step => `<li>${step}</li>`).join('')}
                </ol>
            </div>
        </div>
    `).join('');

    initLucide();
}

function openRecipeDrawer() {
    const drawer = document.getElementById('recipe-drawer');
    if (drawer) {
        drawer.classList.remove('translate-x-full');
    }
}

function closeRecipeDrawer() {
    const drawer = document.getElementById('recipe-drawer');
    if (drawer) {
        drawer.classList.add('translate-x-full');
    }
}

function openRecipeDrawerFor(category) {
    openRecipeDrawer();
}

// ==============================================================================
// 3. DECAY & FRESHNESS TRAJECTORY VIEW (Chart.js)
// ==============================================================================

async function loadDecayTrajectories() {
    try {
        const res = await fetch('/api/v1/decay/trajectories');
        if (!res.ok) return;
        const data = await res.json();
        renderDecayProgressBars(data.trajectories || []);
    } catch (e) {
        console.error('Decay trajectories load error:', e);
    }
}

function renderDecayProgressBars(trajectories) {
    const container = document.getElementById('decay-progress-bars');
    if (!container) return;

    if (trajectories.length === 0) {
        container.innerHTML = `<p class="text-xs text-slate-500 text-center py-4">No trajectory data available.</p>`;
        return;
    }

    container.innerHTML = trajectories.map(t => {
        let barColor = 'bg-emerald-500';
        if (t.current_freshness < 30) barColor = 'bg-rose-500';
        else if (t.current_freshness < 70) barColor = 'bg-amber-500';

        return `
            <div class="glass-card p-3 rounded-xl border border-slate-800/80 mb-3">
                <div class="flex items-center justify-between text-xs font-semibold mb-1.5">
                    <span class="text-white">${t.category}</span>
                    <span class="font-mono text-slate-400">${t.current_freshness}% Fresh • ${t.estimated_days_left}d remaining</span>
                </div>
                <div class="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
                    <div class="h-full rounded-full transition-all duration-500 ${barColor}" style="width: ${t.current_freshness}%"></div>
                </div>
                <div class="flex items-center justify-between text-[10px] text-slate-500 mt-1 font-mono">
                    <span>Decay Point (0%)</span>
                    <span>Critical Alert (30%)</span>
                    <span>Peak Fresh (100%)</span>
                </div>
            </div>
        `;
    }).join('');
}

async function initDecayChart() {
    const ctx = document.getElementById('decay-canvas');
    if (!ctx) return;

    try {
        const res = await fetch('/api/v1/decay/trajectories');
        const data = await res.json();
        const trajectories = data.trajectories || [];

        if (decayChartInstance) {
            decayChartInstance.destroy();
        }

        const colors = ['#10b981', '#f59e0b', '#f43f5e', '#38bdf8', '#a855f7'];

        const datasets = trajectories.slice(0, 5).map((t, idx) => {
            const histPoints = (t.history || []).map((h, i) => ({
                x: `-${(t.history.length - i) * 6}h`,
                y: h.freshness
            }));
            histPoints.push({ x: 'Now', y: t.current_freshness });

            const projPoints = (t.projected || []).map((p, i) => ({
                x: `+${(i + 1) * 12}h`,
                y: p.projected_freshness
            }));

            const allPoints = [...histPoints, ...projPoints];

            return {
                label: `${t.category} (${t.estimated_days_left}d)`,
                data: allPoints,
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                tension: 0.35,
                fill: false,
                borderWidth: 2.5,
                pointRadius: 3
            };
        });

        decayChartInstance = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Freshness: ${ctx.parsed.y}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', callback: v => v + '%' }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Decay chart init error:', e);
    }
}

// ==============================================================================
// 4. IMPACT & ANALYTICS DASHBOARD
// ==============================================================================

async function loadAnalytics() {
    try {
        const res = await fetch('/api/v1/analytics');
        if (!res.ok) return;
        const data = await res.json();

        const elWaste = document.getElementById('impact-waste-reduction');
        const elSavings = document.getElementById('impact-savings');
        const elCO2 = document.getElementById('impact-co2');
        const elRescued = document.getElementById('impact-rescued-count');

        if (elWaste) elWaste.innerText = `${data.food_waste_reduction_percent}%`;
        if (elSavings) elSavings.innerText = `$${data.monthly_grocery_savings_usd.toFixed(2)}`;
        if (elCO2) elCO2.innerText = `${data.co2_reduction_kg} kg`;
        if (elRescued) elRescued.innerText = `${data.total_rescued_items}`;
    } catch (e) {
        console.error('Analytics load error:', e);
    }
}

async function initWeeklyChart() {
    const ctx = document.getElementById('weekly-trends-canvas');
    if (!ctx) return;

    try {
        const res = await fetch('/api/v1/analytics/weekly-trends');
        const data = await res.json();

        if (weeklyChartInstance) {
            weeklyChartInstance.destroy();
        }

        weeklyChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.weeks,
                datasets: [
                    {
                        label: 'Standard Baseline Waste (kg)',
                        data: data.waste_baseline_kg,
                        backgroundColor: 'rgba(244, 63, 94, 0.4)',
                        borderColor: '#f43f5e',
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: 'Edge-Vision Tracked Waste (kg)',
                        data: data.waste_with_edge_vision_kg,
                        backgroundColor: 'rgba(16, 185, 129, 0.5)',
                        borderColor: '#10b981',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', callback: v => v + ' kg' }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Weekly chart init error:', e);
    }
}

// ==============================================================================
// 5. LIVE WEBCAM & AUTO-DETECTION ENGINE
// ==============================================================================

let webcamStream = null;
let autoDetectTimer = null;
let isAutoDetecting = false;
let isProcessingFrame = false;
let autoDetectCount = 0;

async function toggleCameraStream() {
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('cam-placeholder');
    const reticle = document.getElementById('cam-reticle-overlay');
    const btnText = document.getElementById('btn-camera-text');
    const btnSnap = document.getElementById('btn-snap-frame');
    const btnAuto = document.getElementById('btn-toggle-autodetect');
    const statusPill = document.getElementById('cam-status-pill');
    const errorBox = document.getElementById('cam-error-msg');
    const errorText = document.getElementById('cam-error-text');

    if (errorBox) errorBox.classList.add('hidden');

    if (webcamStream) {
        stopCameraStream();
        return;
    }

    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamStream = stream;
        video.srcObject = stream;
        video.classList.remove('hidden');
        if (placeholder) placeholder.classList.add('hidden');
        if (reticle) reticle.classList.remove('hidden');

        btnText.innerText = "Stop Live Camera";
        const camBtn = document.getElementById('btn-toggle-camera');
        if (camBtn) {
            camBtn.classList.remove('bg-emerald-500', 'hover:bg-emerald-400');
            camBtn.classList.add('bg-rose-500', 'hover:bg-rose-400');
        }

        if (btnSnap) btnSnap.disabled = false;
        if (btnAuto) btnAuto.disabled = false;

        statusPill.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span class="text-emerald-400 font-semibold">Live Camera Active</span>
        `;
        statusPill.classList.remove('bg-slate-900', 'text-slate-400', 'border-slate-800');
        statusPill.classList.add('bg-emerald-500/10', 'text-emerald-400', 'border-emerald-500/30');

    } catch (err) {
        console.error("Camera access error:", err);
        if (errorBox && errorText) {
            errorText.innerText = `Camera Error: ${err.message || 'Permission denied or webcam not available.'}`;
            errorBox.classList.remove('hidden');
        }
    }
}

function stopCameraStream() {
    if (isAutoDetecting) {
        stopAutoDetect();
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('cam-placeholder');
    const reticle = document.getElementById('cam-reticle-overlay');
    const btnText = document.getElementById('btn-camera-text');
    const btnSnap = document.getElementById('btn-snap-frame');
    const btnAuto = document.getElementById('btn-toggle-autodetect');
    const statusPill = document.getElementById('cam-status-pill');

    if (video) {
        video.srcObject = null;
        video.classList.add('hidden');
    }
    if (placeholder) placeholder.classList.remove('hidden');
    if (reticle) reticle.classList.add('hidden');

    if (btnText) btnText.innerText = "Start Live Camera";
    const camBtn = document.getElementById('btn-toggle-camera');
    if (camBtn) {
        camBtn.classList.remove('bg-rose-500', 'hover:bg-rose-400');
        camBtn.classList.add('bg-emerald-500', 'hover:bg-emerald-400');
    }

    if (btnSnap) btnSnap.disabled = true;
    if (btnAuto) btnAuto.disabled = true;

    if (statusPill) {
        statusPill.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-slate-600"></span>
            Camera Offline
        `;
        statusPill.classList.remove('bg-emerald-500/10', 'text-emerald-400', 'border-emerald-500/30');
        statusPill.classList.add('bg-slate-900', 'text-slate-400', 'border-slate-800');
    }
}

function toggleAutoDetect() {
    if (isAutoDetecting) {
        stopAutoDetect();
    } else {
        startAutoDetect();
    }
}

function startAutoDetect() {
    if (!webcamStream) return;

    isAutoDetecting = true;
    const btnAuto = document.getElementById('btn-toggle-autodetect');
    const btnText = document.getElementById('autodetect-btn-text');
    const badge = document.getElementById('autodetect-badge');
    const intervalSelect = document.getElementById('autodetect-interval-select');

    const intervalMs = parseInt(intervalSelect ? intervalSelect.value : '2500', 10);

    if (btnText) btnText.innerText = "Stop Auto-Detect";
    if (btnAuto) {
        btnAuto.classList.remove('bg-sky-500/15', 'text-sky-400', 'border-sky-500/30');
        btnAuto.classList.add('bg-rose-500/20', 'text-rose-400', 'border-rose-500/40');
    }
    if (badge) badge.classList.remove('hidden');

    // Run first capture immediately
    captureAndDetectSingleFrame();

    // Start interval loop
    autoDetectTimer = setInterval(() => {
        if (!isProcessingFrame && isAutoDetecting) {
            captureAndDetectSingleFrame();
        }
    }, intervalMs);
}

function stopAutoDetect() {
    isAutoDetecting = false;
    if (autoDetectTimer) {
        clearInterval(autoDetectTimer);
        autoDetectTimer = null;
    }

    const btnAuto = document.getElementById('btn-toggle-autodetect');
    const btnText = document.getElementById('autodetect-btn-text');
    const badge = document.getElementById('autodetect-badge');

    if (btnText) btnText.innerText = "Start Auto-Detect";
    if (btnAuto) {
        btnAuto.classList.remove('bg-rose-500/20', 'text-rose-400', 'border-rose-500/40');
        btnAuto.classList.add('bg-sky-500/15', 'text-sky-400', 'border-sky-500/30');
    }
    if (badge) badge.classList.add('hidden');
}

async function captureAndDetectSingleFrame() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const spinner = document.getElementById('detection-spinner');

    if (!video || !canvas || !webcamStream) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isProcessingFrame = true;
    if (spinner) spinner.classList.remove('hidden');

    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
        if (!blob) throw new Error("Could not create image blob from video canvas");

        const formData = new FormData();
        formData.append('file', blob, 'webcam_live_frame.jpg');
        formData.append('device_id', 'WEBCAM-LIVE-NODE');
        formData.append('temperature_c', (22.5 + (Math.random() * 1.5 - 0.7)).toFixed(1));
        formData.append('humidity_rh', (66.0 + (Math.random() * 4.0 - 2.0)).toFixed(1));

        const res = await fetch('/api/v1/ingest/frame', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const data = await res.json();

        autoDetectCount++;
        const counterEl = document.getElementById('autodetect-counter');
        if (counterEl) counterEl.innerText = `Ingests: ${autoDetectCount}`;

        const timeEl = document.getElementById('latest-frame-time');
        if (timeEl) {
            const now = new Date();
            timeEl.innerText = `${now.toLocaleTimeString()} (${data.detected_count} detected)`;
        }

        const previewEl = document.getElementById('live-annotated-preview');
        if (previewEl && data.annotated_image_url) {
            previewEl.src = data.annotated_image_url + '?t=' + Date.now();
        }

        renderDetectedItemsSummary(data.items || []);

        await Promise.all([
            loadInventory(false),
            loadAnalytics(),
            loadDecayTrajectories()
        ]);

        if (data.recipes_triggered) {
            loadExpiringRecipes();
            const banner = document.getElementById('critical-warning-banner');
            if (banner) banner.classList.remove('hidden');
        }

    } catch (err) {
        console.error("Frame capture/detect error:", err);
    } finally {
        isProcessingFrame = false;
        if (spinner) spinner.classList.add('hidden');
    }
}

function renderDetectedItemsSummary(items) {
    const container = document.getElementById('detected-items-summary');
    if (!container) return;

    if (items.length === 0) {
        container.innerHTML = `
            <div class="text-xs text-slate-500 italic p-2 rounded-lg bg-slate-950/50 border border-slate-900">
                No produce detected in this frame. Point camera closer to produce.
            </div>
        `;
        return;
    }

    container.innerHTML = items.map(item => {
        let badgeClass = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
        if (item.status === 'CRITICAL' || item.freshness_percentage < 30) {
            badgeClass = 'text-rose-400 bg-rose-500/10 border-rose-500/30';
        } else if (item.status === 'APPROACHING_SPOILAGE' || item.freshness_percentage < 70) {
            badgeClass = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
        }

        return `
            <div class="p-2 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs">
                <div class="flex items-center gap-2">
                    <span class="font-bold text-white">${item.category}</span>
                    <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                        YOLO ${item.confidence_score.toFixed(0)}%
                    </span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="font-mono text-slate-300 text-[11px]">${item.estimated_days_left}d left</span>
                    <span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold border ${badgeClass}">
                        ${item.freshness_percentage.toFixed(0)}% Fresh
                    </span>
                </div>
            </div>
        `;
    }).join('');

    initLucide();
}

// ==============================================================================
// 6. ESP32-CAM MOCK STREAM & UPLOAD SIMULATOR
// ==============================================================================

async function handleFrameUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    await uploadFramePayload(file);
}

async function simulateSampleUpload(sampleName) {
    try {
        const sampleUrl = `/static/uploads/${sampleName}`;
        const resp = await fetch(sampleUrl);
        if (!resp.ok) throw new Error('Sample not found');
        const blob = await resp.blob();
        await uploadFramePayload(blob, `sim_${sampleName}`);
    } catch (e) {
        console.error('Simulate upload error:', e);
        alert('Could not load sample frame: ' + e.message);
    }
}

async function uploadFramePayload(fileBlob, filename = 'esp32_frame.jpg') {
    const modal = document.getElementById('ingest-loading-modal');
    if (modal) modal.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('file', fileBlob, filename);
        formData.append('device_id', 'ESP32-CAM-01');
        formData.append('temperature_c', (22.0 + Math.random() * 2.5).toFixed(1));
        formData.append('humidity_rh', (65.0 + Math.random() * 8.0).toFixed(1));

        const res = await fetch('/api/v1/ingest/frame', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error('Ingestion failed');
        const data = await res.json();

        const previewEl = document.getElementById('live-annotated-preview');
        if (previewEl && data.annotated_image_url) {
            previewEl.src = data.annotated_image_url + '?t=' + Date.now();
        }

        renderDetectedItemsSummary(data.items || []);
        await loadDashboardData();

        if (data.recipes_triggered) {
            setTimeout(openRecipeDrawer, 600);
        }
    } catch (err) {
        console.error('Ingest error:', err);
        alert('Ingest error: ' + err.message);
    } finally {
        if (modal) modal.classList.add('hidden');
    }
}

async function triggerTestAmbientAlert() {
    try {
        const res = await fetch('/api/v1/alerts/webhook', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: 'Ambient WhatsApp Test Notification',
                message: 'Edge-Vision alert: 2 items expiring in 48 hours. Recipes generated.',
                channel: 'WhatsApp Webhook'
            })
        });
        const data = await res.json();
        alert('Ambient Alert Dispatched! Check server console log.');
    } catch (e) {
        alert('Alert error: ' + e.message);
    }
}
