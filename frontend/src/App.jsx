import React, { useState, useEffect, useRef } from 'react';

// Deterministic mock metadata generator for React details modal
const generateMockMetadata = (filename, id) => {
  let hash = 0;
  for (let i = 0; i < filename.length; i++) {
    hash = filename.charCodeAt(i) + ((hash << 5) - hash);
  }
  hash = Math.abs(hash);

  const brands = ["AeroStride", "UrbanSole", "ApexAthletics", "NovaForce", "TerraStride", "VoltRun", "ZenSoles"];
  const categories = ["Sport / Running", "Casual Sneaker", "High-Top Basketball", "Outdoor / Trail", "Classic Court", "Minimalist Trainer"];
  const tagsPool = [
    ["Breathable Mesh", "Cushioned Sole", "Ultra-Lightweight", "Active Wear"],
    ["Streetwear", "Premium Leather", "Vulcanized Rubber", "Retro Vibe"],
    ["Ankle Support", "High Grip", "Impact Cushioning", "On-Court Performance"],
    ["Water Resistant", "All-Terrain Tread", "Reinforced Toe", "Heavy Duty"],
    ["Minimalist Design", "Ortholite Sockliner", "Everyday Comfort", "Suede Details"],
    ["Zero Drop", "Flexible Sole", "Barefoot Feel", "Eco-Friendly Material"]
  ];

  const brand = brands[hash % brands.length];
  const categoryIdx = hash % categories.length;
  const category = categories[categoryIdx];
  const price = 69.99 + (hash % 130);
  const tags = tagsPool[categoryIdx];
  const sizes = (6 + (hash % 6)) + " - " + (11 + (hash % 3)) + " (US)";

  return {
    brand,
    category,
    price: `$${price.toFixed(2)}`,
    tags,
    sizes,
    id: `#${id}`
  };
};

function App() {
  // App State variables
  const [catalogStats, setCatalogStats] = useState({ catalogSize: 0, totalFiles: 0, isOutOfSync: false });
  const [samples, setSamples] = useState([]);
  const [results, setResults] = useState([]);
  const [resultsTitle, setResultsTitle] = useState('Catalog Exploration');
  
  const [loading, setLoading] = useState(false);
  const [loaderText, setLoaderText] = useState('Searching database...');
  
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  
  const [modal, setModal] = useState({ isOpen: false, filename: '', id: 0, meta: null });
  const [toasts, setToasts] = useState([]);
  const [queryFile, setQueryFile] = useState(null);
  const [accuracyLevel, setAccuracyLevel] = useState('high');

  const fileInputRef = useRef(null);
  const catalogFileInputRef = useRef(null);

  // Load stats and samples on mount
  useEffect(() => {
    loadCatalogStats();
    loadSamples();
    loadCatalogExploration();
  }, []);

  // Re-trigger search automatically if accuracy level changes during a query
  useEffect(() => {
    if (queryFile) {
      performImageSearch(queryFile);
    }
  }, [accuracyLevel]);

  // Toast Helper
  const showToast = (message, type = 'info') => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts(prev => [...prev, { id, message, type }]);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  };

  // API Call: Fetch Stats
  const loadCatalogStats = async () => {
    try {
      const res = await fetch('/api/catalog/stats');
      if (!res.ok) throw new Error('API offline');
      const data = await res.json();
      setCatalogStats({
        catalogSize: data.catalog_size,
        totalFiles: data.dataset_images_count,
        isOutOfSync: data.is_out_of_sync
      });
    } catch (err) {
      console.error(err);
      showToast('Connection to backend failed.', 'error');
    }
  };

  // API Call: Fetch Preset Samples
  const loadSamples = async () => {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('Failed to fetch samples');
      const data = await res.json();
      setSamples(data);
    } catch (err) {
      console.error(err);
    }
  };

  // API Call: Explore initial catalog cards
  const loadCatalogExploration = async () => {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('Failed to load samples');
      const data = await res.json();
      setResultsTitle('Catalog Exploration');
      setResults(data.map((filename, idx) => ({
        id: idx + 1,
        filename,
        score: null // No score in exploration mode
      })));
    } catch (err) {
      console.error(err);
    }
  };

  // API Call: Image similarity search
  const performImageSearch = async (file) => {
    setLoading(true);
    setLoaderText('Computing image embedding and searching catalog...');
    setResults([]);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('k', 9);

    const thresholdMap = {
      highest: 0.85,
      high: 0.80,
      medium: 0.70,
      relaxed: 0.50,
      none: 0.00
    };
    const minScore = thresholdMap[accuracyLevel] !== undefined ? thresholdMap[accuracyLevel] : 0.80;

    try {
      const res = await fetch(`/api/search/image?k=9&min_score=${minScore}`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Visual search failed. Check backend logs.');
      const data = await res.json();
      
      setResultsTitle('Visually similar shoes');
      setResults(data.results);
    } catch (err) {
      showToast(err.message, 'error');
      setResultsTitle('Visual Search');
    } finally {
      setLoading(false);
    }
  };

  // Helper: process local files
  const processSelectedFile = (file) => {
    if (!file.type.startsWith('image/')) {
      showToast('Please upload a valid image file.', 'error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewSrc(e.target.result);
      setQueryFile(file);
      performImageSearch(file);
    };
    reader.readAsDataURL(file);
  };

  // Dropzone drag/drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  };

  // Preset click handler
  const handleSelectSample = async (filename) => {
    setLoading(true);
    setLoaderText('Loading sample image...');
    try {
      const res = await fetch(`/dataset/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error('Failed to retrieve sample image');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type });
      
      setPreviewSrc(`/dataset/${encodeURIComponent(filename)}`);
      setQueryFile(file);
      performImageSearch(file);
    } catch (err) {
      showToast(err.message, 'error');
      setLoading(false);
    }
  };

  // Recursive search trigger
  const handleTriggerVisualSearchFromCard = async (e, filename) => {
    e.stopPropagation();
    setLoading(true);
    setLoaderText(`Searching matching visuals for ${filename}...`);
    try {
      const res = await fetch(`/dataset/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error('Failed to retrieve catalog image');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type });

      setPreviewSrc(`/dataset/${encodeURIComponent(filename)}`);
      setQueryFile(file);
      performImageSearch(file);
      showToast(`Visual search triggered for: ${filename}`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
      setLoading(false);
    }
  };

  // Clear query preview
  const handleClearImageQuery = () => {
    setPreviewSrc(null);
    setQueryFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    loadCatalogExploration();
  };

  // API Call: Upload new image
  const handleUploadShoeImage = async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    if (!file.type.startsWith('image/')) {
      showToast('Only image files are allowed in dataset.', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showToast('Uploading shoe image to dataset...', 'warning');

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Upload failed. Image format might not be supported.');
      const data = await res.json();
      
      showToast(data.message, 'success');
      loadCatalogStats();
      loadSamples();
      loadCatalogExploration();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      if (catalogFileInputRef.current) catalogFileInputRef.current.value = '';
    }
  };

  // API Call: Re-index catalog
  const handleReindexCatalog = async () => {
    showToast('Running in-memory catalog re-indexing. Please wait...', 'warning');
    
    // Simulate updating in stats display
    setCatalogStats(prev => ({ ...prev, isOutOfSync: true }));

    try {
      const res = await fetch('/api/reindex', { method: 'POST' });
      if (!res.ok) throw new Error('Re-indexing failed.');
      const data = await res.json();
      
      showToast(data.message, 'success');
      loadCatalogStats();
      loadCatalogExploration();
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  // Modal handlers
  const handleOpenModal = (filename, id) => {
    const meta = generateMockMetadata(filename, id);
    setModal({
      isOpen: true,
      filename,
      id,
      meta
    });
  };

  const handleCloseModal = () => {
    setModal({ isOpen: false, filename: '', id: 0, meta: null });
  };

  const handleExploreSimilarFromModal = () => {
    const filename = modal.filename;
    handleCloseModal();
    
    // Create synthetic event
    const mockEvent = { stopPropagation: () => {} };
    handleTriggerVisualSearchFromCard(mockEvent, filename);
  };

  return (
    <div className="app-container">
      {/* Header Panel */}
      <header>
        <div className="header-title">
          <h1>Shoe Visual Search</h1>
          <p>Locate matches in your catalog using DINOv2 visual features and similarity indexing.</p>
        </div>
        <div className="status-panel">
          <span className={`status-indicator ${catalogStats.isOutOfSync ? 'warning' : ''}`}></span>
          <span>
            {catalogStats.isOutOfSync ? 'Index out of sync' : `Catalog: ${catalogStats.catalogSize} shoes`}
          </span>
        </div>
      </header>

      {/* Main Grid layout */}
      <div className="workspace">
        
        {/* Left Side: Sidebar */}
        <div className="sidebar">
          
          {/* Visual Finder Search Box */}
          <div className="panel">
            <div className="panel-title">Visual Finder</div>
            
            <div className="form-group">
              <div 
                className={`dropzone ${isDragOver ? 'dragover' : ''}`}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {!previewSrc ? (
                  <div className="dropzone-default">
                    <div className="dropzone-icon">📷</div>
                    <div className="dropzone-text">Drop query image, or <span>browse</span></div>
                    <input 
                      type="file" 
                      ref={fileInputRef}
                      className="file-input" 
                      accept="image/*" 
                      onChange={handleFileSelect}
                    />
                  </div>
                ) : (
                  <div className="preview-container" onClick={e => e.stopPropagation()}>
                    <img className="image-preview" src={previewSrc} alt="Query preview" />
                    <button type="button" className="btn-clear-image" onClick={handleClearImageQuery}>Remove Image</button>
                  </div>
                )}
              </div>

              {/* Accuracy Selector */}
              <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <span className="samples-title">Comparison Accuracy</span>
                <select 
                  className="text-input" 
                  value={accuracyLevel} 
                  onChange={(e) => setAccuracyLevel(e.target.value)}
                >
                  <option value="highest">Highest (85% Match & Above)</option>
                  <option value="high">High (80% Match & Above)</option>
                  <option value="medium">Medium (70% Match & Above)</option>
                  <option value="relaxed">Relaxed (50% Match & Above)</option>
                  <option value="none">Show All Matches (No Threshold)</option>
                </select>
              </div>

              {/* Preset Samples */}
              <div className="samples-section" style={{ marginTop: '1.5rem' }}>
                <div className="samples-title">Quick Test Samples</div>
                <div className="samples-grid">
                  {samples.map((sample, idx) => (
                    <img 
                      key={idx}
                      src={`/dataset/${encodeURIComponent(sample)}`}
                      className="sample-thumb"
                      title={sample}
                      alt={sample}
                      onClick={() => handleSelectSample(sample)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Catalog Dashboard Panel */}
          <div className="panel">
            <div className="panel-title">Catalog Administration</div>
            
            <div className="stats-list">
              <div className="stats-item">
                <span className="stats-label">Indexed Shoes:</span>
                <span className="stats-value">{catalogStats.catalogSize}</span>
              </div>
              <div className="stats-item">
                <span className="stats-label">Files in Dataset:</span>
                <span className="stats-value">{catalogStats.totalFiles}</span>
              </div>
            </div>

            {catalogStats.isOutOfSync && (
              <div className="warning-banner">
                <span>⚠</span>
                <div>Index is out of sync. Please click Re-index to update the catalog database.</div>
              </div>
            )}

            <button 
              className="btn-secondary" 
              style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }} 
              onClick={handleReindexCatalog}
            >
              <span>🔄</span> Re-index Catalog
            </button>

            {/* Catalog Upload Form */}
            <div className="catalog-uploader">
              <h4>Add New Shoe to Catalog</h4>
              <div 
                className="catalog-dropzone" 
                onClick={() => catalogFileInputRef.current && catalogFileInputRef.current.click()}
              >
                Click or Drop photo here
                <input 
                  type="file" 
                  ref={catalogFileInputRef}
                  className="file-input" 
                  accept="image/*" 
                  onChange={handleUploadShoeImage}
                />
              </div>
            </div>
          </div>

        </div>

        {/* Right Side: Results Area */}
        <div className="main-content">
          <div className="content-header">
            <h3 className="content-title">{resultsTitle}</h3>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
              {results.length > 0 && `${results.length} matches`}
            </span>
          </div>

          <div className="results-grid">
            {/* Loading Indicator */}
            {loading && (
              <div className="loader">
                <div className="spinner"></div>
                <p>{loaderText}</p>
              </div>
            )}

            {/* Render Matches List */}
            {!loading && results.map((item, idx) => (
              <div 
                key={idx}
                className="result-card" 
                onClick={() => handleOpenModal(item.filename, item.id)}
              >
                <div className="image-wrapper">
                  <img 
                    className="result-image" 
                    src={`/dataset/${encodeURIComponent(item.filename)}`} 
                    alt={item.filename} 
                    loading="lazy" 
                  />
                </div>
                <div className="result-info">
                  <div className="card-top">
                    {item.score !== null ? (
                      <span className="score-badge">Match: {(item.score * 100).toFixed(1)}%</span>
                    ) : (
                      <span className="score-badge" style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: 'var(--text-secondary)' }}>Item</span>
                    )}
                  </div>
                  <div className="filename-text" title={item.filename}>{item.filename}</div>
                  <div className="card-footer">
                    <span className="result-id">ID: {item.id}</span>
                    <button 
                      className="btn-card-action" 
                      onClick={(e) => handleTriggerVisualSearchFromCard(e, item.filename)}
                    >
                      Find Similar
                    </button>
                  </div>
                </div>
              </div>
            ))}

            {/* Empty State */}
            {!loading && results.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">👟</div>
                <p>Drop a shoe photo or select a quick test sample to begin visual search.</p>
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Details Popup Modal */}
      {modal.isOpen && (
        <div className="modal" onClick={handleCloseModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <span className="close-btn" onClick={handleCloseModal}>&times;</span>
            <div className="modal-grid">
              <div className="modal-image-wrapper">
                <img src={`/dataset/${encodeURIComponent(modal.filename)}`} alt="Shoe Detail" />
              </div>
              <div className="modal-info">
                <h2>{modal.filename}</h2>
                <div className="spec-list">
                  <div className="spec-item">
                    <span className="spec-label">Brand:</span>
                    <span className="spec-value">{modal.meta.brand}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Category:</span>
                    <span className="spec-value">{modal.meta.category}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Estimated Price:</span>
                    <span className="spec-value">{modal.meta.price}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Sizes Available:</span>
                    <span className="spec-value">{modal.meta.sizes}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Catalog ID:</span>
                    <span className="spec-value">{modal.meta.id}</span>
                  </div>
                </div>
                <div className="style-tags">
                  {modal.meta.tags.map((tag, idx) => (
                    <span key={idx} className="tag">{tag}</span>
                  ))}
                </div>
                <div className="modal-actions">
                  <button className="btn-primary" onClick={handleExploreSimilarFromModal}>
                    🔍 Find Visually Similar
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Custom Context Toasts */}
      <div className="toast-container">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span>
              {toast.type === 'success' && '✓'}
              {toast.type === 'error' && '✕'}
              {toast.type === 'warning' && '⏳'}
              {toast.type === 'info' && 'ℹ'}
              &nbsp; {toast.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
