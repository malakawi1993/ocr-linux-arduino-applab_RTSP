/**
 * Edge Impulse OCR - Camera control and real-time status updates
 */

// Update interval in milliseconds
const UPDATE_INTERVAL = 100;

// State tracking
let isRunning = false;
let statusInterval = null;

/**
 * Fetch available cameras from the server
 */
async function refreshCameras() {
    const select = document.getElementById('camera-select');
    select.innerHTML = '<option value="">Scanning cameras...</option>';
    document.getElementById('start-btn').disabled = true;

    try {
        const response = await fetch('/cameras');
        const data = await response.json();

        select.innerHTML = '';

        if (data.cameras.length === 0) {
            select.innerHTML = '<option value="">No cameras found</option>';
            return;
        }

        data.cameras.forEach(cam => {
            const option = document.createElement('option');
            option.value = cam.index;
            option.textContent = cam.name;
            select.appendChild(option);
        });

        document.getElementById('start-btn').disabled = false;

    } catch (error) {
        select.innerHTML = '<option value="">Error loading cameras</option>';
        showError('Failed to load camera list: ' + error.message);
    }
}

/**
 * Start the inference engine
 */
async function startInference() {
    const rtspUrl = document.getElementById('rtsp-url').value.trim();
    const cameraIndex = document.getElementById('camera-select').value;

    // RTSP URL takes priority; otherwise fall back to the selected wired camera
    if (rtspUrl === '' && cameraIndex === '') {
        showError('Please enter an RTSP URL or select a camera first');
        return;
    }

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const btnText = document.getElementById('start-btn-text');

    // Show loading state
    startBtn.disabled = true;
    btnText.innerHTML = 'Starting...';
    hideError();

    try {
        const startBody = rtspUrl !== ''
            ? { source: rtspUrl }
            : { camera: parseInt(cameraIndex) };

        const response = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(startBody)
        });

        const data = await response.json();

        if (data.success) {
            isRunning = true;
            startBtn.style.display = 'none';
            stopBtn.style.display = 'flex';

            // Show video feed
            document.getElementById('video-placeholder').style.display = 'none';
            const videoFeed = document.getElementById('video-feed');
            videoFeed.src = '/video_feed?' + Date.now();  // Cache bust
            videoFeed.style.display = 'block';

            // Update status indicator
            document.getElementById('status-dot').classList.add('running');
            document.getElementById('status-text').textContent = 'Running';

            // Start polling for status
            statusInterval = setInterval(updateStatus, UPDATE_INTERVAL);

            // Update model info
            document.getElementById('detector-name').textContent = data.detector_info || 'Loaded';
            document.getElementById('predictor-name').textContent = data.predictor_info || 'Loaded';

        } else {
            showError(data.error || 'Failed to start inference');
            startBtn.disabled = false;
            btnText.textContent = 'Start';
        }

    } catch (error) {
        showError('Failed to start inference: ' + error.message);
        startBtn.disabled = false;
        btnText.textContent = 'Start';
    }
}

/**
 * Stop the inference engine
 */
async function stopInference() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const btnText = document.getElementById('start-btn-text');

    try {
        const response = await fetch('/stop', { method: 'POST' });
        const data = await response.json();

        isRunning = false;

        // Stop status polling
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }

        // Hide video feed
        const videoFeed = document.getElementById('video-feed');
        videoFeed.src = '';
        videoFeed.style.display = 'none';
        document.getElementById('video-placeholder').style.display = 'block';

        // Update UI
        stopBtn.style.display = 'none';
        startBtn.style.display = 'flex';
        startBtn.disabled = false;
        btnText.textContent = 'Start';

        // Update status indicator
        document.getElementById('status-dot').classList.remove('running');
        document.getElementById('status-text').textContent = 'Stopped';

        // Reset stats
        document.getElementById('fps-value').textContent = '--';
        document.getElementById('detections-value').textContent = '0';
        document.getElementById('detector-time').textContent = '--';
        document.getElementById('predictor-time').textContent = '--';
        document.getElementById('timing-info').textContent = '--';
        document.getElementById('results-list').innerHTML =
            '<div class="no-results">No detections yet</div>';

    } catch (error) {
        showError('Failed to stop inference: ' + error.message);
    }
}

/**
 * Fetch and display current inference status
 */
async function updateStatus() {
    if (!isRunning) return;

    try {
        const response = await fetch('/status');
        const data = await response.json();

        if (!data.running) {
            // Inference stopped unexpectedly
            stopInference();
            return;
        }

        // Update statistics
        document.getElementById('fps-value').textContent = data.fps.toFixed(1);
        document.getElementById('detections-value').textContent = data.results.length;
        document.getElementById('detector-time').textContent = data.detector_ms;
        document.getElementById('predictor-time').textContent = data.predictor_ms;
        document.getElementById('timing-info').textContent =
            `Frame ${data.frame_count} | Total: ${data.detector_ms + data.predictor_ms}ms`;

        // Update results list
        const resultsList = document.getElementById('results-list');
        if (data.results.length > 0) {
            resultsList.innerHTML = data.results.map(r => `
                <div class="result-item">
                    <div class="result-text">${escapeHtml(r.text)}</div>
                    <div class="result-meta">
                        ${(r.confidence * 100).toFixed(1)}% confidence
                    </div>
                </div>
            `).join('');
        } else {
            resultsList.innerHTML = '<div class="no-results">No text detected</div>';
        }

    } catch (error) {
        console.error('Status update failed:', error);
    }
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
}

/**
 * Hide error message
 */
function hideError() {
    document.getElementById('error-message').classList.remove('show');
}

/**
 * Escape HTML to prevent XSS attacks
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Prefill the RTSP URL field with the server's default stream (if any)
    fetch('/config')
        .then(r => r.json())
        .then(data => {
            if (data.default_rtsp_url) {
                document.getElementById('rtsp-url').value = data.default_rtsp_url;
            }
        })
        .catch(() => {});

    refreshCameras();
});
