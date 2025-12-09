// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const loadingDiv = document.getElementById('loading');
const resultDiv = document.getElementById('result');
const initialDiv = document.getElementById('initialState');

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#3498db';
    uploadArea.style.backgroundColor = '#f8f9fa';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#ddd';
    uploadArea.style.backgroundColor = 'white';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ddd';
    uploadArea.style.backgroundColor = 'white';
    
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// File Input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

// Handle File
function handleFile(file) {
    // نمایش loading
    showLoading();
    
    // نمایش preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // ارسال به سرور
    uploadFile(file);
}

// Upload File
function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        showResult(data);
    })
    .catch(error => {
        hideLoading();
        showError('Connection error: ' + error.message);
    });
}

// Show Loading
function showLoading() {
    initialDiv.style.display = 'none';
    resultDiv.style.display = 'none';
    loadingDiv.style.display = 'block';
}

// Hide Loading
function hideLoading() {
    loadingDiv.style.display = 'none';
}

// Show Result
function showResult(data) {
    resultDiv.style.display = 'block';
    
    // تنظیم رنگ بر اساس سرطان بودن
    const mainResult = document.getElementById('mainResult');
    if (data.is_cancer) {
        mainResult.className = 'alert alert-danger cancer-alert';
        mainResult.innerHTML = `
            <h4><i class="fas fa-exclamation-triangle"></i> سرطان شناسایی شد</h4>
            <p class="mb-2">${data.main_description}</p>
        `;
    } else {
        mainResult.className = 'alert alert-success';
        mainResult.innerHTML = `
            <h4><i class="fas fa-check-circle"></i> ضایعه خوش‌خیم</h4>
            <p class="mb-2">${data.main_description}</p>
        `;
    }
    
    // اطمینان و ریسک
    document.getElementById('confidence').textContent = `${data.main_confidence.toFixed(1)}%`;
    document.getElementById('riskLevel').textContent = data.risk_level;
    document.getElementById('riskLevel').className = `risk-badge risk-${data.risk_level.toLowerCase().includes('high') ? 'high' : 
                                                      data.risk_level.toLowerCase().includes('medium') ? 'medium' : 'low'}`;
    
    // توصیه
    document.getElementById('recommendation').textContent = data.recommendation;
    
    // سایر احتمالات
    const otherPredictions = document.getElementById('otherPredictions');
    otherPredictions.innerHTML = '';
    
    data.all_predictions.forEach((pred, index) => {
        if (index > 0) { // بعد از اولین مورد
            const predDiv = document.createElement('div');
            predDiv.className = 'mb-3 p-3 border rounded';
            predDiv.innerHTML = `
                <div class="d-flex justify-content-between">
                    <div>
                        <strong>${pred.class.toUpperCase()}</strong>
                        <div class="text-muted small">${pred.description}</div>
                    </div>
                    <div class="text-end">
                        <div class="badge bg-info">${pred.confidence.toFixed(1)}%</div>
                        <div class="small mt-1">${pred.risk_level}</div>
                    </div>
                </div>
                <div class="progress mt-2" style="height: 10px;">
                    <div class="progress-bar bg-info" style="width: ${pred.confidence}%"></div>
                </div>
            `;
            otherPredictions.appendChild(predDiv);
        }
    });
    
    // زمان
    document.getElementById('timestamp').textContent = data.timestamp;
}

// Show Error
function showError(message) {
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <div class="alert alert-danger">
            <h4><i class="fas fa-exclamation-circle"></i> خطا</h4>
            <p>${message}</p>
            <button class="btn btn-primary mt-3" onclick="location.reload()">
                <i class="fas fa-redo"></i> تلاش مجدد
            </button>
        </div>
    `;
}

// Use Sample Image
function useSampleImage(imgElement) {
    const imgUrl = imgElement.src;
    
    // دانلود تصویر نمونه
    fetch(imgUrl)
        .then(response => response.blob())
        .then(blob => {
            const file = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
            handleFile(file);
        });
}

// Model Info
function loadModelInfo() {
    fetch('/api/model_info')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'loaded') {
                document.getElementById('modelStatus').innerHTML = `
                    <span class="badge bg-success">Loaded</span>
                    <small class="text-muted ms-2">${data.model_type} - ${(data.accuracy * 100).toFixed(1)}% accuracy</small>
                `;
            }
        });
}

// Load on page ready
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
});