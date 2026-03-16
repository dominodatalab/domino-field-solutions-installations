// Triton Admin Dashboard JavaScript utilities

/**
 * Change the current namespace and reload the page
 */
function changeNamespace(namespace) {
    const url = new URL(window.location.href);
    url.searchParams.set('namespace', namespace);
    window.location.href = url.toString();
}

/**
 * Format bytes to human readable string
 */
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Format duration in seconds to human readable string
 */
function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return 'N/A';
    if (seconds < 60) return Math.floor(seconds) + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.floor(seconds % 60) + 's';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
}

/**
 * Format ISO timestamp to relative time
 */
function formatRelativeTime(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    const now = new Date();
    const diff = (now - date) / 1000;

    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
    return date.toLocaleDateString();
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Confirm a dangerous action
 */
function confirmAction(message) {
    return window.confirm(message);
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard', 'success');
    } catch (err) {
        console.error('Failed to copy:', err);
        showToast('Failed to copy', 'error');
    }
}

// Add toast styles dynamically
const toastStyles = document.createElement('style');
toastStyles.textContent = `
    .toast {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        border-radius: 4px;
        color: white;
        font-size: 14px;
        z-index: 1000;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
    }

    .toast.show {
        opacity: 1;
        transform: translateY(0);
    }

    .toast-info {
        background-color: #2563eb;
    }

    .toast-success {
        background-color: #16a34a;
    }

    .toast-error {
        background-color: #dc2626;
    }

    .toast-warning {
        background-color: #d97706;
    }
`;
document.head.appendChild(toastStyles);
