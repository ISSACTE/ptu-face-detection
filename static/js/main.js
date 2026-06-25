// Loading overlay for form submissions
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', function() {
    const btn = this.querySelector('[type=submit]');
    if (btn && !btn.dataset.noLoader) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    }
  });
});

// Auto-dismiss alerts
setTimeout(() => {
  document.querySelectorAll('.alert-dismissible').forEach(el => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
    if (bsAlert) bsAlert.close();
  });
}, 5000);
