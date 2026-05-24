import { useState, useCallback, useEffect, useRef } from 'react';
import { predictHeartDisease } from './services/api';

const INITIAL_FORM = {
  age: '', sex: 'M', chestPainType: 'ATA', restingBP: '',
  cholesterol: '', fastingBS: '0', restingECG: 'Normal',
  maxHR: '', exerciseAngina: 'N', oldpeak: '', stSlope: 'Flat',
};

/* ── Animated Heart SVG ── */
function HeartIcon({ size = 24, animate = false }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={animate ? 'heartbeat-icon' : ''}>
      <path
        d="M12 21.593c-5.63-5.539-11-10.297-11-14.402C1 3.748 3.498 2 6 2c1.977 0 3.884.996 5 2.686C12.116 2.996 14.023 2 16 2c2.502 0 5 1.748 5 5.191 0 4.105-5.37 8.863-11 14.402z"
        fill="currentColor"
      />
    </svg>
  );
}

/* ── ECG Line SVG ── */
function EcgLine() {
  return (
    <svg viewBox="0 0 400 60" className="ecg-line" preserveAspectRatio="none">
      <polyline
        points="0,30 60,30 75,10 90,50 105,30 160,30 170,5 180,55 190,30 250,30 260,15 270,45 280,30 340,30 355,8 370,52 380,30 400,30"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ── Toggle Chip ── */
function ToggleGroup({ options, value, onChange }) {
  return (
    <div className="chip-group">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          className={`chip ${value === opt.value ? 'chip-active' : ''}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/* ── Toast ── */
function Toast({ message, type, onClose }) {
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(onClose, 4500);
    return () => clearTimeout(t);
  }, [message, onClose]);

  if (!message) return null;
  return (
    <div className={`toast toast-${type}`}>
      <span className="toast-icon">{type === 'success' ? '✓' : '!'}</span>
      <span>{message}</span>
    </div>
  );
}

/* ── Step Indicator ── */
function StepIndicator({ current, total }) {
  return (
    <div className="step-indicator">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} className={`step-dot ${i < current ? 'done' : i === current ? 'active' : ''}`} />
      ))}
    </div>
  );
}

/* ── Animated Gauge ── */
function RiskGauge({ percentage, isHighRisk }) {
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const [offset, setOffset] = useState(circumference);

  useEffect(() => {
    const t = setTimeout(() => {
      setOffset(circumference - (percentage / 100) * circumference);
    }, 200);
    return () => clearTimeout(t);
  }, [percentage, circumference]);

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 180 180" className="gauge-svg">
        <defs>
          <linearGradient id="g-danger" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#c0392b" />
            <stop offset="100%" stopColor="#e74c3c" />
          </linearGradient>
          <linearGradient id="g-safe" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#16a085" />
            <stop offset="100%" stopColor="#27ae60" />
          </linearGradient>
        </defs>
        <circle className="gauge-track" cx="90" cy="90" r={radius} />
        <circle
          className="gauge-fill"
          cx="90" cy="90" r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          stroke={isHighRisk ? 'url(#g-danger)' : 'url(#g-safe)'}
        />
      </svg>
      <div className="gauge-center">
        <div className={`gauge-pct ${isHighRisk ? 'pct-danger' : 'pct-safe'}`}>{percentage}%</div>
        <div className="gauge-sub">probability</div>
      </div>
    </div>
  );
}

/* ── Result Card ── */
function ResultDisplay({ result, onReset }) {
  const isHigh = result.prediction === 'HIGH RISK';
  const prob = result.probability;

  return (
    <div className={`result-wrap ${isHigh ? 'result-high' : 'result-low'}`}>
      {/* Status banner */}
      <div className={`result-banner ${isHigh ? 'banner-high' : 'banner-low'}`}>
        <span className="banner-icon">{isHigh ? '⚠' : '✓'}</span>
        <div>
          <div className="banner-title">{isHigh ? 'Elevated Risk Detected' : 'Low Risk'}</div>
          <div className="banner-sub">AI Confidence: {result.confidence}%</div>
        </div>
      </div>

      <RiskGauge percentage={prob} isHighRisk={isHigh} />

      <p className="result-note">
        {isHigh
          ? `Your clinical profile indicates a ${prob}% probability of heart disease. Please consult a cardiologist for a thorough evaluation.`
          : `Your clinical profile shows only a ${prob}% probability of heart disease. Continue maintaining a healthy lifestyle.`}
      </p>

      {/* Probability breakdown */}
      <div className="prob-section">
        <div className="prob-label-row">
          <span>Heart Disease</span>
          <span className="prob-val prob-danger">{result.details.heart_disease_prob}%</span>
        </div>
        <div className="prob-track">
          <div className="prob-fill fill-danger" style={{ width: `${result.details.heart_disease_prob}%` }} />
        </div>

        <div className="prob-label-row" style={{ marginTop: 12 }}>
          <span>Healthy</span>
          <span className="prob-val prob-safe">{result.details.healthy_prob}%</span>
        </div>
        <div className="prob-track">
          <div className="prob-fill fill-safe" style={{ width: `${result.details.healthy_prob}%` }} />
        </div>
      </div>

      <p className="disclaimer">
        This is an AI-assisted screening tool — not a medical diagnosis. Always consult a qualified physician.
      </p>

      <button className="btn-reset" onClick={onReset}>
        ← Run Another Assessment
      </button>
    </div>
  );
}

/* ══════════════════════════════════
   MAIN APP
══════════════════════════════════ */
export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [toast, setToast] = useState({ message: '', type: 'info' });
  const formRef = useRef(null);

  const update = (name, value) => setForm((p) => ({ ...p, [name]: value }));
  const showToast = (message, type = 'info') => setToast({ message, type });

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    try {
      const res = await predictHeartDisease(form);
      setResult(res);
      showToast('Analysis complete', 'success');
    } catch (err) {
      const msg = err.response?.data?.error
        || 'Could not connect. Ensure the Flask backend is running on port 5000.';
      showToast(msg, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [form]);

  const handleReset = () => {
    setResult(null);
    setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 50);
  };

  return (
    <div className="app">

      {/* ── Ambient background ── */}
      <div className="ambient" aria-hidden="true">
        <div className="amb-blob amb-1" />
        <div className="amb-blob amb-2" />
        <div className="amb-blob amb-3" />
      </div>

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-mark">
              <HeartIcon size={18} />
            </div>
            <span className="logo-name">HeartSense<span className="logo-ai"> AI</span></span>
          </div>
          <div className="header-ecg" aria-hidden="true">
            <EcgLine />
          </div>
          <div className="header-badge">Clinical Screening Tool</div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">

        {/* Hero */}
        <div className="hero">
          <div className="hero-eyebrow">Cardiovascular Risk Assessment</div>
          <h1 className="hero-h1">
            Know Your Heart's<br />Health Profile
          </h1>
          <p className="hero-p">
            Enter your clinical measurements and our machine-learning model will
            estimate your probability of cardiovascular disease.
          </p>
        </div>

        {!result ? (
          /* ─── FORM ─── */
          <div className="form-card" ref={formRef}>

            {/* Trust signals */}
            <div className="trust-row">
              <span className="trust-pill">🔒 Private — not stored</span>
              <span className="trust-pill">🤖 ML-powered</span>
              <span className="trust-pill">⚡ Instant results</span>
            </div>

            <form onSubmit={handleSubmit} noValidate>

              {/* ── Section: Personal ── */}
              <div className="section-hd">
                <div className="section-num">01</div>
                <div>
                  <div className="section-title">Personal Information</div>
                  <div className="section-sub">Basic demographic details</div>
                </div>
              </div>

              <div className="fields-grid">
                <div className="field">
                  <label className="field-label">Age</label>
                  <div className="input-wrap">
                    <input
                      type="number"
                      className="field-input"
                      placeholder="e.g. 55"
                      value={form.age}
                      onChange={(e) => update('age', e.target.value)}
                      min="1" max="120" required
                    />
                    <span className="input-unit">yrs</span>
                  </div>
                </div>

                <div className="field">
                  <label className="field-label">Biological Sex</label>
                  <ToggleGroup
                    options={[{ value: 'M', label: 'Male' }, { value: 'F', label: 'Female' }]}
                    value={form.sex}
                    onChange={(v) => update('sex', v)}
                  />
                </div>

                <div className="field field-full">
                  <label className="field-label">Chest Pain Type</label>
                  <select
                    className="field-input field-select"
                    value={form.chestPainType}
                    onChange={(e) => update('chestPainType', e.target.value)}
                  >
                    <option value="TA">Typical Angina (TA)</option>
                    <option value="ATA">Atypical Angina (ATA)</option>
                    <option value="NAP">Non-Anginal Pain (NAP)</option>
                    <option value="ASY">Asymptomatic (ASY)</option>
                  </select>
                </div>
              </div>

              <div className="section-divider" />

              {/* ── Section: Vitals ── */}
              <div className="section-hd">
                <div className="section-num">02</div>
                <div>
                  <div className="section-title">Vital Signs</div>
                  <div className="section-sub">Measurements from your last check-up</div>
                </div>
              </div>

              <div className="fields-grid">
                <div className="field">
                  <label className="field-label">Resting Blood Pressure</label>
                  <div className="input-wrap">
                    <input
                      type="number"
                      className="field-input"
                      placeholder="e.g. 130"
                      value={form.restingBP}
                      onChange={(e) => update('restingBP', e.target.value)}
                      min="50" max="300" required
                    />
                    <span className="input-unit">mmHg</span>
                  </div>
                </div>

                <div className="field">
                  <label className="field-label">Cholesterol</label>
                  <div className="input-wrap">
                    <input
                      type="number"
                      className="field-input"
                      placeholder="e.g. 240"
                      value={form.cholesterol}
                      onChange={(e) => update('cholesterol', e.target.value)}
                      min="50" max="700" required
                    />
                    <span className="input-unit">mg/dl</span>
                  </div>
                </div>

                <div className="field">
                  <label className="field-label">Fasting Blood Sugar &gt;120 mg/dl</label>
                  <ToggleGroup
                    options={[{ value: '0', label: 'No  (≤120)' }, { value: '1', label: 'Yes (>120)' }]}
                    value={form.fastingBS}
                    onChange={(v) => update('fastingBS', v)}
                  />
                </div>

                <div className="field">
                  <label className="field-label">Resting ECG</label>
                  <select
                    className="field-input field-select"
                    value={form.restingECG}
                    onChange={(e) => update('restingECG', e.target.value)}
                  >
                    <option value="Normal">Normal</option>
                    <option value="ST">ST-T Wave Abnormality</option>
                    <option value="LVH">Left Ventricular Hypertrophy</option>
                  </select>
                </div>
              </div>

              <div className="section-divider" />

              {/* ── Section: Cardiac ── */}
              <div className="section-hd">
                <div className="section-num">03</div>
                <div>
                  <div className="section-title">Cardiac Metrics</div>
                  <div className="section-sub">Stress test and exercise data</div>
                </div>
              </div>

              <div className="fields-grid">
                <div className="field">
                  <label className="field-label">Max Heart Rate</label>
                  <div className="input-wrap">
                    <input
                      type="number"
                      className="field-input"
                      placeholder="e.g. 150"
                      value={form.maxHR}
                      onChange={(e) => update('maxHR', e.target.value)}
                      min="50" max="250" required
                    />
                    <span className="input-unit">bpm</span>
                  </div>
                </div>

                <div className="field">
                  <label className="field-label">Exercise-Induced Angina</label>
                  <ToggleGroup
                    options={[{ value: 'N', label: 'No' }, { value: 'Y', label: 'Yes' }]}
                    value={form.exerciseAngina}
                    onChange={(v) => update('exerciseAngina', v)}
                  />
                </div>

                <div className="field">
                  <label className="field-label">Oldpeak — ST Depression</label>
                  <div className="input-wrap">
                    <input
                      type="number"
                      className="field-input"
                      placeholder="e.g. 1.5"
                      value={form.oldpeak}
                      onChange={(e) => update('oldpeak', e.target.value)}
                      step="0.1" min="-5" max="10" required
                    />
                    <span className="input-unit">mm</span>
                  </div>
                </div>

                <div className="field">
                  <label className="field-label">ST Slope</label>
                  <select
                    className="field-input field-select"
                    value={form.stSlope}
                    onChange={(e) => update('stSlope', e.target.value)}
                  >
                    <option value="Up">Upsloping ↗</option>
                    <option value="Flat">Flat →</option>
                    <option value="Down">Downsloping ↘</option>
                  </select>
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                className="btn-submit"
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="loading-state">
                    <span className="spinner" />
                    Analyzing your data…
                  </span>
                ) : (
                  <span className="submit-inner">
                    <HeartIcon size={18} />
                    Assess My Risk
                  </span>
                )}
              </button>

            </form>
          </div>
        ) : (
          <ResultDisplay result={result} onReset={handleReset} />
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <div className="footer-inner">
          <span>HeartSense AI</span>
          <span className="footer-dot">·</span>
          <span>For educational purposes only — not a substitute for medical advice</span>
        </div>
      </footer>

      {/* ── Toast ── */}
      <Toast
        message={toast.message}
        type={toast.type}
        onClose={() => setToast({ message: '', type: 'info' })}
      />
    </div>
  );
}