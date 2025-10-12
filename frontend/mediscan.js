// ========== CONFIGURATION ==========
const API_BASE_URL = 'http://localhost:8000'; // Your FastAPI server
const FRONTEND_API_KEY = 'mediscan_secret_key_2024'; // Only for frontend -> backend

// ========== STATE ==========
let selectedFile = null;

// ========== DOM ELEMENTS ==========
const form = document.getElementById('diagnosisForm');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const submitBtn = document.getElementById('submitBtn');
const resultsContainer = document.getElementById('resultsContainer');

// ========== FILE UPLOAD ==========
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        uploadArea.classList.add('has-file');

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadContent.innerHTML = `
                <img src="${e.target.result}" alt="Preview" class="preview-image">
                <p style="color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem;">${file.name}</p>
            `;
        };
        reader.readAsDataURL(file);
    }
});

// ========== FORM SUBMISSION ==========
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const symptoms = document.getElementById('symptoms').value.trim();
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const medicalHistory = document.getElementById('medicalHistory').value.trim();

    if (!symptoms) {
        showError('Please describe your symptoms');
        return;
    }

    setLoading(true);
    showLoadingState();

    try {
        const formData = new FormData();
        formData.append('symptoms', symptoms);
        if (age) formData.append('age', age);
        if (gender) formData.append('gender', gender);
        if (medicalHistory) formData.append('medical_history', medicalHistory);
        if (selectedFile) formData.append('image', selectedFile);

        const response = await fetch(`${API_BASE_URL}/api/v2/diagnose`, {
            method: 'POST',
            headers: { 'x-api-key': FRONTEND_API_KEY }, // Frontend key only
            body: formData
        });

        if (!response.ok) {
    let errorMessage = 'Diagnosis failed';
    try {
        const errorData = await response.json();
        if (errorData.detail) {
            errorMessage = errorData.detail;
        } else if (errorData.error) {
            errorMessage = errorData.error;
        }
    } catch (error) {
    console.error('Diagnosis error:', error);

    // Try to parse JSON if it's an object
    let errorMsg = '';
    if (error instanceof Error) {
        errorMsg = error.message;
    } else if (typeof error === 'object') {
        errorMsg = JSON.stringify(error);
    } else {
        errorMsg = String(error);
    }

    showError(errorMsg || 'An error occurred. Please try again.');
} finally {
    setLoading(false);
}

}

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Diagnosis error:', error);
        showError(error.message || 'An error occurred. Please try again.');
    } finally {
        setLoading(false);
    }
});

// ========== UI FUNCTIONS ==========
function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.innerHTML = isLoading
        ? `<svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg><span>Analyzing...</span>`
        : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/><path d="M12 6v6l4 2"/></svg><span>Get Diagnosis</span>`;
}

function showLoadingState() {
    resultsContainer.innerHTML = `
        <div class="loading-container">
            <svg class="loading-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
            </svg>
            <p style="color: #6b7280;">Analyzing with AI...</p>
        </div>
    `;
}

function showError(message) {
    if (typeof message === 'object') message = JSON.stringify(message);
    resultsContainer.innerHTML = `
        <div class="alert alert-error">
            <div>
                <p style="font-weight: 500;">Error</p>
                <p style="font-size: 0.875rem; margin-top: 0.25rem;">${escapeHtml(message)}</p>
            </div>
        </div>
    `;
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function displayResults(result) {
    resultsContainer.innerHTML = `
        <div class="diagnosis-header">
            <p class="section-title">Primary Diagnosis</p>
            <h3 class="diagnosis-title">${escapeHtml(result.primary_diagnosis)}</h3>
            <span class="risk-badge ${result.risk_level.toLowerCase()}">${escapeHtml(result.risk_level)}</span>
            <div class="diagnosis-meta">
                <div>Confidence: ${result.confidence_score}%</div>
                ${result.icd10_code ? `<div>ICD-10: ${escapeHtml(result.icd10_code)}</div>` : ''}
            </div>
            ${result.human_reasoning ? `
                <div class="human-reasoning" style="margin-top: 1rem; font-size: 0.95rem; color: #374151;">
                    <strong>AI Explanation:</strong>
                    <p>${escapeHtml(result.human_reasoning)}</p>
                </div>
            ` : ''}
        </div>
    `;
}


console.log('AI MediScan Pro - Frontend Loaded');
