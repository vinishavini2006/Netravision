/**
 * NetraVision - Explainable AI Retinal Screening Interactive Engine
 * Handles patient intake, fundus upload, Grad-CAM split slider, and clinical reporting.
 */

// Global state
const state = {
    selectedFile: null,
    selectedSample: null,
    currentEye: 'OD',
    screeningResult: null,
    activeViewMode: 'split', // 'split', 'side-by-side', 'heatmap', 'enhanced'
    splitPercent: 50,
    isDraggingSplit: false,
    currentColormap: 'jet',
    currentAlpha: 0.55
};

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    fetchSystemStatus();
    loadSamples();
    loadRecentScreenings();
    generateInitialPatientId();
    setupSplitSliderEvents();
});

function initElements() {
    // Eye buttons
    document.querySelectorAll('.eye-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.eye-btn').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            state.currentEye = e.currentTarget.dataset.eye;
        });
    });

    // File input & dropzone
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fundusFileInput');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Run Screening button
    document.getElementById('btnRunScreening').addEventListener('click', executeScreening);

    // View mode tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
            e.currentTarget.classList.add('active');
            setViewMode(e.currentTarget.dataset.mode);
        });
    });

    // Colormap & Opacity controls
    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) {
        colormapSelect.addEventListener('change', (e) => {
            state.currentColormap = e.target.value;
            if (state.selectedFile || state.selectedSample) {
                executeScreening();
            }
        });
    }

    const opacitySlider = document.getElementById('opacitySlider');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', (e) => {
            state.currentAlpha = parseFloat(e.target.value);
            document.getElementById('opacityValue').textContent = `${Math.round(state.currentAlpha * 100)}%`;
        });
        opacitySlider.addEventListener('change', () => {
            if (state.selectedFile || state.selectedSample) {
                executeScreening();
            }
        });
    }

    // Print button
    const btnPrint = document.getElementById('btnPrintReport');
    if (btnPrint) {
        btnPrint.addEventListener('click', () => {
            if (state.screeningResult && state.screeningResult.screening_id) {
                window.open(`/api/report/${state.screeningResult.screening_id}`, '_blank');
            } else {
                alert('Please run a screening diagnosis first.');
            }
        });
    }
}

// Generate new Patient ID
function generateInitialPatientId() {
    const year = new Date().getFullYear();
    const rand = Math.floor(1000 + Math.random() * 9000);
    const pidField = document.getElementById('patientId');
    if (pidField && !pidField.value) {
        pidField.value = `NV-PHC-${year}-${rand}`;
    }
}

// System Status Check
async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/system-status');
        const data = await res.json();
        const statusText = document.getElementById('systemStatusText');
        if (statusText) {
            statusText.textContent = `${data.system} (${data.device.toUpperCase()})`;
        }
    } catch (e) {
        console.warn('System status check offline:', e);
    }
}

// Handle Image File Selection
function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        alert('Please select a valid image file (JPEG or PNG).');
        return;
    }
    state.selectedFile = file;
    state.selectedSample = null;

    // Deselect sample buttons
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (e) => {
        updateImagePreview(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
}

// Load built-in clinical samples
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        const container = document.getElementById('samplesGrid');
        if (!container) return;
        container.innerHTML = '';

        data.samples.forEach(sample => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.title = sample.description;
            card.innerHTML = `
                <img src="/api/samples/${sample.filename}" class="sample-thumb" alt="${sample.title}" />
                <div class="sample-badge">${sample.badge}</div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                selectSample(sample);
            });
            container.appendChild(card);
        });
    } catch (e) {
        console.error('Error loading sample scans:', e);
    }
}

async function selectSample(sample) {
    state.selectedSample = sample;
    state.selectedFile = null;

    // Fetch sample image as blob
    try {
        const response = await fetch(`/api/samples/${sample.filename}`);
        const blob = await response.blob();
        state.selectedFile = new File([blob], sample.filename, { type: 'image/jpeg' });
        
        const previewUrl = URL.createObjectURL(blob);
        updateImagePreview(previewUrl, sample.title);
    } catch (e) {
        console.error('Error selecting sample:', e);
    }
}

function updateImagePreview(src, title) {
    const previewContainer = document.getElementById('imagePreviewArea');
    const previewImg = document.getElementById('previewImg');
    const previewTitle = document.getElementById('previewTitle');

    if (previewContainer && previewImg) {
        previewImg.src = src;
        if (previewTitle) previewTitle.textContent = title || 'Fundus Image Loaded';
        previewContainer.style.display = 'block';
    }

    // Set initial split slider views as well
    const imgScan = document.getElementById('sliderOriginalImg');
    const imgOverlay = document.getElementById('sliderOverlayImg');
    if (imgScan) imgScan.src = src;
    if (imgOverlay) imgOverlay.src = src;

    // Enable Run Screening button
    document.getElementById('btnRunScreening').disabled = false;
}

// Execute AI Diagnostic Screening Pipeline
async function executeScreening() {
    if (!state.selectedFile) {
        alert('Please select or upload a retinal fundus scan first.');
        return;
    }

    showLoading(true, "Preprocessing Retinal FOV...", "Centering fundus circle and assessing optical sharpness...");

    const formData = new FormData();
    formData.append('image', state.selectedFile);
    formData.append('patient_id', document.getElementById('patientId').value || '');
    formData.append('full_name', document.getElementById('patientName').value || 'Anonymous Patient');
    formData.append('age', document.getElementById('patientAge').value || '50');
    formData.append('gender', document.getElementById('patientGender').value || 'Unspecified');
    formData.append('eye', state.currentEye);
    formData.append('colormap', state.currentColormap);
    formData.append('alpha', state.currentAlpha);

    try {
        // Step update in loader
        setTimeout(() => updateLoadingStep("Running EfficientNet CNN...", "Classifying microaneurysms and exudate patterns..."), 500);
        setTimeout(() => updateLoadingStep("Hooking Feature Gradients...", "Computing Explainable AI Grad-CAM heatmaps..."), 1100);

        const res = await fetch('/api/screen', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Screening failed.');
        }

        const data = await res.json();
        state.screeningResult = data;
        renderResults(data);
        loadRecentScreenings();

    } catch (e) {
        alert(`Screening Error: ${e.message}`);
        console.error(e);
    } finally {
        showLoading(false);
    }
}

// Render Diagnostic Output Dashboard
function renderResults(data) {
    // 1. Severity Banner
    const banner = document.getElementById('severityBanner');
    banner.className = `severity-banner grade-${data.predicted_grade}`;
    
    document.getElementById('severityGradeTitle').textContent = `Grade ${data.predicted_grade}: ${data.grade_label}`;
    document.getElementById('severityCategoryLabel').textContent = `${data.eye === 'OD' ? 'Right Eye (OD)' : 'Left Eye (OS)'} • ${data.severity} Risk`;
    document.getElementById('severityDescription').textContent = data.clinical_protocol.summary;
    document.getElementById('confidenceScoreNumber').textContent = `${Math.round(data.confidence_score * 100)}%`;

    // 2. Explainability Images in Split View
    document.getElementById('sliderOriginalImg').src = data.images.original;
    document.getElementById('sliderOverlayImg').src = data.images.gradcam_overlay;
    document.getElementById('sideBySideOriginal').src = data.images.original;
    document.getElementById('sideBySideHeatmap').src = data.images.gradcam_overlay;
    document.getElementById('rawHeatmapImg').src = data.images.raw_heatmap;
    document.getElementById('claheEnhancedImg').src = data.images.clahe_enhanced;

    // 3. Probabilities Distribution
    const probContainer = document.getElementById('probabilityBars');
    if (probContainer && data.probabilities) {
        probContainer.innerHTML = '';
        const gradeLabels = ["Grade 0: Normal", "Grade 1: Mild NPDR", "Grade 2: Moderate NPDR", "Grade 3: Severe NPDR", "Grade 4: Proliferative DR"];
        const colors = ["#10b981", "#3b82f6", "#f59e0b", "#ea580c", "#ef4444"];

        for (let g = 0; g < 5; g++) {
            const prob = (data.probabilities[g] || 0) * 100;
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <div class="prob-header">
                    <span class="prob-name" style="color: ${g === data.predicted_grade ? colors[g] : '#f8fafc'}">${gradeLabels[g]}</span>
                    <span class="prob-val">${prob.toFixed(1)}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-bar" style="width: ${prob}%; background: ${colors[g]};"></div>
                </div>
            `;
            probContainer.appendChild(item);
        }
    }

    // 4. Lesions Table
    const lesionBody = document.getElementById('lesionsTableBody');
    if (lesionBody) {
        lesionBody.innerHTML = '';
        if (data.lesions && data.lesions.length > 0) {
            data.lesions.forEach(l => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${l.region_id}</strong></td>
                    <td><span class="intensity-badge" style="background:rgba(14,165,233,0.15); color:#0ea5e9;">${l.quadrant}</span></td>
                    <td><span class="intensity-badge">${Math.round(l.attention_intensity * 100)}%</span></td>
                    <td style="color:#94a3b8; font-size:11px;">${l.clinical_significance}</td>
                `;
                lesionBody.appendChild(tr);
            });
        } else {
            lesionBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:#64748b; padding:16px;">No focal capillary lesion hotspots detected.</td></tr>`;
        }
    }

    // 5. Clinical Protocol Directives
    const protocolList = document.getElementById('clinicalActionList');
    if (protocolList && data.clinical_protocol) {
        protocolList.innerHTML = '';
        data.clinical_protocol.primary_care_plan.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            protocolList.appendChild(li);
        });
    }

    // 6. QA Score
    if (data.qa_metrics) {
        const qaBadge = document.getElementById('qaMetricsBadge');
        if (qaBadge) {
            qaBadge.textContent = `Quality: ${data.qa_metrics.quality_grade} (Sharpness: ${data.qa_metrics.sharpness_score})`;
            qaBadge.style.color = data.qa_metrics.is_diagnostic_quality ? '#10b981' : '#f59e0b';
        }
    }

    // Scroll smoothly to results
    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// View Mode Switcher
function setViewMode(mode) {
    state.activeViewMode = mode;
    const splitContainer = document.getElementById('splitSliderContainer');
    const sideBySideContainer = document.getElementById('sideBySideContainer');
    const rawHeatmapContainer = document.getElementById('rawHeatmapContainer');
    const claheContainer = document.getElementById('claheContainer');

    // Hide all
    if (splitContainer) splitContainer.style.display = 'none';
    if (sideBySideContainer) sideBySideContainer.style.display = 'none';
    if (rawHeatmapContainer) rawHeatmapContainer.style.display = 'none';
    if (claheContainer) claheContainer.style.display = 'none';

    if (mode === 'split' && splitContainer) {
        splitContainer.style.display = 'block';
    } else if (mode === 'side-by-side' && sideBySideContainer) {
        sideBySideContainer.style.display = 'grid';
    } else if (mode === 'heatmap' && rawHeatmapContainer) {
        rawHeatmapContainer.style.display = 'block';
    } else if (mode === 'enhanced' && claheContainer) {
        claheContainer.style.display = 'block';
    }
}

// Setup Interactive Split Curtain Slider
function setupSplitSliderEvents() {
    const container = document.getElementById('splitSliderContainer');
    const divider = document.getElementById('splitDivider');
    const overlayLayer = document.getElementById('layerOverlay');

    if (!container || !divider || !overlayLayer) return;

    function updateSlider(clientX) {
        const rect = container.getBoundingClientRect();
        let x = clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        const percent = (x / rect.width) * 100;
        state.splitPercent = percent;

        divider.style.left = `${percent}%`;
        overlayLayer.style.clipPath = `polygon(0 0, ${percent}% 0, ${percent}% 100%, 0 100%)`;
    }

    divider.addEventListener('mousedown', (e) => {
        state.isDraggingSplit = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
        state.isDraggingSplit = false;
    });

    window.addEventListener('mousemove', (e) => {
        if (!state.isDraggingSplit) return;
        updateSlider(e.clientX);
    });

    // Touch support for clinic tablets
    divider.addEventListener('touchstart', () => { state.isDraggingSplit = true; });
    window.addEventListener('touchend', () => { state.isDraggingSplit = false; });
    window.addEventListener('touchmove', (e) => {
        if (!state.isDraggingSplit || !e.touches[0]) return;
        updateSlider(e.touches[0].clientX);
    });

    // Click anywhere on container to snap slider
    container.addEventListener('click', (e) => {
        if (e.target !== divider && !divider.contains(e.target)) {
            updateSlider(e.clientX);
        }
    });
}

// Load Recent Screening Archive
async function loadRecentScreenings() {
    try {
        const res = await fetch('/api/screenings?limit=8');
        const data = await res.json();
        const tbody = document.getElementById('screeningsArchiveBody');
        if (!tbody) return;
        tbody.innerHTML = '';

        if (!data.screenings || data.screenings.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#64748b; padding:16px;">No previous screenings in this session.</td></tr>`;
            return;
        }

        data.screenings.forEach(s => {
            const tr = document.createElement('tr');
            const dateStr = s.created_at ? s.created_at.split(' ')[0] : '-';
            tr.innerHTML = `
                <td><code>${s.patient_id}</code></td>
                <td><strong>${s.full_name || 'Patient'}</strong> (${s.age}y/${s.gender ? s.gender[0] : '-'})</td>
                <td><span class="intensity-badge" style="background:rgba(14,165,233,0.1); color:#38bdf8;">${s.eye}</span></td>
                <td><strong>Grade ${s.predicted_grade}</strong> (${s.grade_label.split(' ')[0]})</td>
                <td><span class="intensity-badge" style="background:rgba(16,185,129,0.15); color:#10b981;">${Math.round(s.confidence_score * 100)}%</span></td>
                <td><a href="/api/report/${s.screening_id}" target="_blank" class="btn-secondary" style="padding:4px 10px; font-size:11px;">📄 Report</a></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn('Error loading screenings history:', e);
    }
}

// Loading Modal Helpers
function showLoading(active, title, sub) {
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) return;
    if (active) {
        overlay.classList.add('active');
        updateLoadingStep(title, sub);
    } else {
        overlay.classList.remove('active');
    }
}

function updateLoadingStep(title, sub) {
    const titleEl = document.getElementById('loadingTitle');
    const subEl = document.getElementById('loadingSub');
    if (titleEl && title) titleEl.textContent = title;
    if (subEl && sub) subEl.textContent = sub;
}
