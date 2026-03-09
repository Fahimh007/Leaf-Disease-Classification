document.addEventListener('DOMContentLoaded', function(){
  const input = document.getElementById('image-input');
  const preview = document.getElementById('preview-img');
  const form = document.getElementById('predict-form');
  const loader = document.getElementById('loader-overlay');
  const resetBtn = document.getElementById('reset-btn');
  const dropArea = document.getElementById('drop-area');
  const openCameraBtn = document.getElementById('open-camera-btn');
  const cameraVideo = document.getElementById('camera-video');
  const captureCanvas = document.getElementById('capture-canvas');
  const captureBtn = document.getElementById('capture-btn');
  let stream = null;
  const urlBtn = document.getElementById('url-btn');
  const urlModal = document.getElementById('urlModal');
  const urlInput = document.getElementById('url-input');
  const urlLoadBtn = document.getElementById('url-load-btn');
  const urlAlert = document.getElementById('url-alert');
  const pasteModal = document.getElementById('pasteModal');
  const pasteImg = document.getElementById('paste-img');
  const pasteAlert = document.getElementById('paste-alert');
  const pasteUseBtn = document.getElementById('paste-use-btn');
  let pastedBlob = null;

  // add loaded class when image finishes loading for smooth transition
  if(preview){
    preview.addEventListener('load', function(){ preview.classList.add('loaded'); });
    // If image already loaded (cached), ensure loaded class is present
    if(preview.complete && preview.naturalWidth && preview.naturalWidth > 0){
      preview.classList.add('loaded');
    }
  }

  input.addEventListener('change', function(e){
    const file = e.target.files[0];
    if(!file) return;
    const reader = new FileReader();
    reader.onload = function(ev){
      preview.classList.remove('loaded');
      preview.src = ev.target.result;
    }
    reader.readAsDataURL(file);
    // clear fetched marker when user selects a local file
    const fetchedInput = document.getElementById('fetched-input');
    if(fetchedInput) fetchedInput.value = '';
  });

  // Drag & drop
  if(dropArea){
    ;['dragenter','dragover'].forEach(evt=> dropArea.addEventListener(evt, e=>{e.preventDefault(); dropArea.classList.add('dragover')}));
    ;['dragleave','drop'].forEach(evt=> dropArea.addEventListener(evt, e=>{e.preventDefault(); dropArea.classList.remove('dragover')}));
    dropArea.addEventListener('drop', function(e){
      const file = e.dataTransfer.files[0];
      if(!file) return;
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      const reader = new FileReader();
      reader.onload = function(ev){ preview.classList.remove('loaded'); preview.src = ev.target.result; }
      reader.readAsDataURL(file);
      // clear fetched marker
      const fetchedInput = document.getElementById('fetched-input');
      if(fetchedInput) fetchedInput.value = '';
    });
  }

  // Camera handling
  function startCamera(){
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      .then(s => { stream = s; cameraVideo.srcObject = s; })
      .catch(err => { console.warn('Camera not available', err); });
  }

  function stopCamera(){
    if(stream){
      stream.getTracks().forEach(t=>t.stop());
      stream = null;
      cameraVideo.srcObject = null;
    }
  }

  const cameraModal = document.getElementById('cameraModal');
  if(cameraModal){
    cameraModal.addEventListener('shown.bs.modal', startCamera);
    cameraModal.addEventListener('hidden.bs.modal', stopCamera);
  }

  if(captureBtn){
    captureBtn.addEventListener('click', function(){
      if(!cameraVideo || cameraVideo.readyState < 2) return;
      const w = cameraVideo.videoWidth;
      const h = cameraVideo.videoHeight;
      captureCanvas.width = w; captureCanvas.height = h;
      const ctx = captureCanvas.getContext('2d');
      ctx.drawImage(cameraVideo, 0, 0, w, h);
        captureCanvas.toBlob(function(blob){
        const filename = 'capture.png';
        const file = new File([blob], filename, { type: 'image/png' });
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        const reader = new FileReader();
        reader.onload = function(ev){ preview.classList.remove('loaded'); preview.src = ev.target.result; }
        reader.readAsDataURL(blob);
        // clear fetched marker when using camera
        const fetchedInput = document.getElementById('fetched-input');
        if(fetchedInput) fetchedInput.value = '';
        // close modal
        const modal = bootstrap.Modal.getInstance(cameraModal);
        modal.hide();
      }, 'image/png');
    });
  }

  // URL import
  function showUrlAlert(message){
    if(!urlAlert) return;
    urlAlert.textContent = message;
    urlAlert.classList.remove('d-none');
  }

  function clearUrlAlert(){
    if(!urlAlert) return;
    urlAlert.classList.add('d-none');
    urlAlert.textContent = '';
  }

  if(urlLoadBtn){
    urlLoadBtn.addEventListener('click', async function(){
      clearUrlAlert();
      const url = (urlInput && urlInput.value || '').trim();
      if(!url){ showUrlAlert('Please enter an image URL.'); return; }
      try{
        // ask server to fetch the image (avoids browser CORS)
        const res = await fetch('/fetch_url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });
        const data = await res.json();
        if(!data.ok){
          showUrlAlert(data.error || 'Server failed to fetch image');
          return;
        }

        // clear any file input and set hidden fetched filename
        input.value = '';
        const fetchedInput = document.getElementById('fetched-input');
        if(fetchedInput) fetchedInput.value = data.filename;

          // set preview to server-served URL (use absolute URL to avoid relative path issues)
          preview.classList.remove('loaded');
          const fileUrl = (data.url && (data.url.startsWith('http://') || data.url.startsWith('https://'))) ? data.url : (window.location.origin + data.url);

          // preload image with a temporary Image() to avoid race conditions
          const temp = new Image();
          temp.onload = function(){
            // set preview src only after successful preload
            preview.classList.remove('loaded');
            preview.src = fileUrl;
            // hide modal and allow the user to click Upload & Predict manually
            const modal = bootstrap.Modal.getInstance(urlModal) || new bootstrap.Modal(urlModal);
            modal.hide();
            // Do NOT auto-submit: user should review preview and click predict
          };
          temp.onerror = function(){ showUrlAlert('Failed to load preview image.'); };
          temp.src = fileUrl;
      }catch(err){
        console.error(err);
        showUrlAlert('Could not load image from URL.');
      }
    });
  }

  // Paste image clipboard handling
  function clearPasteAlert(){ if(pasteAlert) { pasteAlert.classList.add('d-none'); pasteAlert.textContent=''; } }

  async function handlePasteEvent(e){
    clearPasteAlert();
    pastedBlob = null;
    const items = (e.clipboardData && e.clipboardData.items) || [];
    for(let i=0;i<items.length;i++){
      const it = items[i];
      if(it.kind === 'file' && it.type.startsWith('image/')){
        const blob = it.getAsFile();
        pastedBlob = blob;
        const url = URL.createObjectURL(blob);
        if(pasteImg) pasteImg.src = url;
        e.preventDefault();
        return;
      }
    }
    if(pasteAlert) { pasteAlert.textContent = 'No image found in clipboard.'; pasteAlert.classList.remove('d-none'); }
  }

  if(pasteModal){
    pasteModal.addEventListener('shown.bs.modal', function(){
      // attach paste listener
      document.addEventListener('paste', handlePasteEvent);
      pastedBlob = null;
      if(pasteImg) pasteImg.src = pasteImg.getAttribute('src');
      clearPasteAlert();
    });
    pasteModal.addEventListener('hidden.bs.modal', function(){
      document.removeEventListener('paste', handlePasteEvent);
    });
  }

  if(pasteUseBtn){
    pasteUseBtn.addEventListener('click', function(){
      if(!pastedBlob){ if(pasteAlert) { pasteAlert.textContent = 'No pasted image to use.'; pasteAlert.classList.remove('d-none'); } return; }
      const filename = 'pasted.png';
      const file = new File([pastedBlob], filename, { type: pastedBlob.type });
      const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
      // clear fetched marker
      const fetchedInput = document.getElementById('fetched-input'); if(fetchedInput) fetchedInput.value = '';
      // update preview & close modal
      const reader = new FileReader(); reader.onload = function(ev){ preview.classList.remove('loaded'); preview.src = ev.target.result; }
      reader.readAsDataURL(pastedBlob);
      const modal = bootstrap.Modal.getInstance(pasteModal); if(modal) modal.hide();
      // auto-submit
      setTimeout(()=>{ form.requestSubmit ? form.requestSubmit() : form.submit(); }, 200);
    });
  }

  form.addEventListener('submit', function(){
    // show loader overlay while uploading/predicting
    loader.classList.remove('d-none');
  });

  // Animate progress bars smoothly when results are rendered
  function animateProgressBars(){
    const bars = document.querySelectorAll('.prob-list .progress-bar');
    bars.forEach((bar, i) => {
      const raw = bar.getAttribute('data-prob') || bar.getAttribute('aria-valuenow') || '0';
      const prob = Math.max(0, Math.min(100, parseFloat(raw)));
      // ensure start at 0 for animation
      bar.style.width = '0%';
      setTimeout(() => {
        bar.style.width = prob + '%';
        bar.setAttribute('aria-valuenow', prob);
      }, 200 + i * 100);
    });
  }

  // run animation on load (useful after server-rendered results)
  animateProgressBars();

  resetBtn.addEventListener('click', function(){
    input.value = '';
    preview.classList.remove('loaded');
    preview.src = preview.getAttribute('data-default') || preview.src;
    const fetchedInput = document.getElementById('fetched-input');
    if(fetchedInput) fetchedInput.value = '';
    // hide loader if shown
    loader.classList.add('d-none');
    // remove results (simple approach: reload page to clear server-side results)
    window.location.href = window.location.pathname;
  });
});
