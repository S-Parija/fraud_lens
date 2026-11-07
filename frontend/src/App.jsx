import { useEffect, useRef, useState } from 'react';
import { requestJson } from './api.js';

const PAYMENT_FIELDS = [
  { name: 'amount', label: 'Amount (USD)', type: 'number', step: '0.01' },
  { name: 'timestamp', label: 'Date & time', type: 'datetime-local' },
  { name: 'merchant_category', label: 'Type of business', type: 'select' },
  { name: 'merchant_country', label: 'Merchant country', type: 'country' },
  { name: 'device_type', label: 'Payment method', type: 'select' },
  { name: 'card_present', label: 'Physical card used?', type: 'binary' },
];

const CONTEXT_GROUPS = [
  {
    title: 'Security checks',
    fields: [
      { name: 'ip_risk_score', label: 'Connection risk (0–100)', type: 'number', max: 100, step: '0.1', help: 'Higher numbers mean a less trusted connection.' },
      { name: 'device_known', label: 'Device used before?', type: 'binary' },
      { name: 'has_2fa', label: 'Extra verification enabled?', type: 'binary' },
      { name: 'is_foreign_txn', label: 'International payment?', type: 'binary' },
    ],
  },
  {
    title: 'Account activity',
    fields: [
      { name: 'velocity_1h', label: 'Payments in the last hour', type: 'number', step: '1' },
      { name: 'time_since_last_s', label: 'Seconds since previous payment', type: 'number', step: 'any' },
      { name: 'amount_vs_avg_ratio', label: 'Compared with usual spending', type: 'number', step: 'any', help: '1 = usual amount; 3 = three times the usual amount.' },
      { name: 'account_age_days', label: 'Account age (days)', type: 'number', step: 'any' },
    ],
  },
];
const ALL_FIELDS = [...PAYMENT_FIELDS, ...CONTEXT_GROUPS.flatMap((group) => group.fields)];

const FRIENDLY_NAMES = {
  atm: 'ATM', pos_terminal: 'Card terminal', web_browser: 'Online purchase',
  mobile_app: 'Mobile app', phone_ivr: 'Phone payment', gas_station: 'Fuel station', online_retail: 'Online shopping',
};
const SAMPLE_LABELS = { original: 'Card payment', unfamiliar: 'Online purchase', unusual: 'Rapid payments' };
const CHECK_NAMES = {
  'Amount and deviation': 'Amount compared with usual spending',
  Velocity: 'Recent payment activity', 'IP risk': 'Connection security',
  Device: 'Device familiarity', Authentication: 'Extra verification',
  'Merchant geography': 'Payment location', 'Merchant category': 'Type of business',
  'Card channel': 'How the card was used', 'Account age': 'Account history', Time: 'Time of payment',
};
const RISK_MESSAGES = {
  Low: { label: 'Low risk', title: 'No strong warning signs', action: 'Continue with your usual transaction checks.' },
  Review: { label: 'Needs review', title: 'Take a closer look', action: 'Review the payment details and confirm anything unusual with the customer.' },
  High: { label: 'High risk', title: 'Verify this payment', action: 'Verify the payment with the customer before proceeding.' },
};
const countryNames = new Intl.DisplayNames(['en'], { type: 'region' });
const localTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

function humanize(value) {
  return FRIENDLY_NAMES[value] ?? value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function percent(value) {
  return `${(value * 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: value < 0.001 ? 4 : 2 })}%`;
}

/** Show the sample in local time while retaining its exact instant. */
function sampleToForm(transaction) {
  const date = new Date(transaction.timestamp);
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return { ...transaction, timestamp: localDate.toISOString().slice(0, 19) };
}

/** Include the sample's context even while its optional editor is collapsed. */
function formToTransaction(form) {
  const transaction = { ...form, timestamp: new Date(form.timestamp).toISOString() };
  for (const field of ALL_FIELDS) {
    if (field.type === 'number' || field.type === 'binary') {
      if (form[field.name] === '') throw new Error(`Please enter ${field.label.toLowerCase()}.`);
      transaction[field.name] = Number(form[field.name]);
    }
  }
  transaction.merchant_country = transaction.merchant_country.toUpperCase();
  return transaction;
}

function Field({ field, value, options, onChange }) {
  const help = field.type === 'datetime-local' ? `Your local time · ${localTimezone}` : field.help;
  const common = {
    id: field.name, name: field.name, value: value ?? '', required: true,
    'aria-describedby': help ? `${field.name}-help` : undefined,
    onChange: (event) => onChange(field.name, event.target.value),
  };
  const isSelect = ['select', 'country', 'binary'].includes(field.type);
  let choices = [];
  if (field.type === 'binary') {
    choices = [{ value: 1, label: 'Yes' }, { value: 0, label: 'No' }];
  } else if (isSelect) {
    choices = (options[field.name] ?? []).map((item) => ({
      value: item, label: field.type === 'country' ? countryNames.of(item) : humanize(item),
    }));
    if (value && !choices.some((choice) => choice.value === value)) {
      choices.push({ value, label: field.type === 'country' ? countryNames.of(value) : humanize(value) });
    }
  }
  return <div className="field">
    <label htmlFor={field.name}>{field.label}</label>
    {isSelect
      ? <select {...common}>{choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}</select>
      : <input {...common} type={field.type} min={field.type === 'number' ? 0 : undefined} max={field.max} step={field.type === 'datetime-local' ? 1 : field.step} />}
    {help && <span id={`${field.name}-help`} className="field-help">{help}</span>}
  </div>;
}

/** Use plain labels for the strongest checks, without technical attribution values. */
function keyChecks(explanation) {
  return [...(explanation?.risk_increasing_characteristics ?? []), ...(explanation?.risk_decreasing_characteristics ?? [])]
    .filter((item) => CHECK_NAMES[item.characteristic_group])
    .sort((a, b) => Math.abs(b.calibrated_logodds_effect) - Math.abs(a.calibrated_logodds_effect))
    .slice(0, 3);
}

function Results({ result, stale, busy }) {
  const heading = useRef(null);
  useEffect(() => {
    if (result) {
      heading.current?.focus({ preventScroll: true });
      heading.current?.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
    }
  }, [result]);

  if (!result) return <aside className="result-panel empty-result" aria-label="Transaction result">
    <span className="result-icon" aria-hidden="true">✓</span>
    <h2>Your result will appear here</h2>
    <p>Review the payment details, then select <strong>Check transaction</strong>.</p>
  </aside>;

  const message = RISK_MESSAGES[result.risk_level];
  const checks = keyChecks(result.explanation);
  return <aside className={`result-panel ${stale ? 'is-stale' : ''}`} aria-label="Transaction result" aria-busy={busy}>
    <div className="result-heading"><h2 ref={heading} tabIndex={-1}>Check result</h2><span className={`risk-badge ${result.risk_level.toLowerCase()}`}>{message.label}</span></div>
    {stale && <p className="stale-notice" role="status">Details changed. Check the transaction again for an updated result.</p>}
    <p className="score-label">Estimated chance of fraud</p>
    <div className="probability">{percent(result.fraud_probability)}</div>
    <div className="probability-track" role="meter" aria-label="Chance of fraud" aria-valuemin={0} aria-valuemax={100} aria-valuenow={result.fraud_probability * 100}>
      <span className={result.risk_level.toLowerCase()} style={{ width: `${result.fraud_probability * 100}%` }} />
    </div>
    <section className="next-step"><h3>{message.title}</h3><p>{message.action}</p></section>
    {checks.length > 0 && <details className="check-details">
      <summary>Why this result?</summary>
      <ul>{checks.map((check) => <li key={check.characteristic_group}>
        <span>{CHECK_NAMES[check.characteristic_group]}</span>
        <span className={`check-direction ${check.calibrated_logodds_effect > 0 ? 'higher' : 'lower'}`}>{check.calibrated_logodds_effect > 0 ? 'Raises concern' : 'Lowers concern'}</span>
      </li>)}</ul>
      <p className="field-help">Factors that raised or lowered concern for this payment.</p>
    </details>}
    <p className="result-footnote">An estimate to support your review, not a confirmation of fraud.</p>
  </aside>;
}

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [form, setForm] = useState({});
  const [selectedSample, setSelectedSample] = useState('original');
  const [result, setResult] = useState(null);
  const [stale, setStale] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const revision = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    requestJson('/api/metadata', { signal: controller.signal })
      .then((data) => {
        setMetadata(data);
        setForm(sampleToForm(data.samples[0].transaction));
        setSelectedSample(data.samples[0].id);
      })
      .catch((failure) => { if (failure.name !== 'AbortError') setError(failure.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [retryCount]);

  function updateField(name, value) {
    revision.current += 1;
    setForm((previous) => ({ ...previous, [name]: value }));
    setSelectedSample('custom');
    setStale(true);
    setError('');
  }

  function loadSample(id) {
    const sample = metadata.samples.find((item) => item.id === id);
    revision.current += 1;
    setForm(sampleToForm(sample.transaction));
    setSelectedSample(id);
    setStale(true);
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    const submittedRevision = revision.current;
    try {
      const assessment = await requestJson('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formToTransaction(form)),
      });
      setResult(assessment);
      setStale(revision.current !== submittedRevision);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  // Reveal invalid context values even if the employee closed the editor after editing.
  function revealInvalidField(event) {
    const details = event.target.closest('details');
    if (details) details.open = true;
  }

  return <>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="/" aria-label="FraudLens home"><span className="brand-symbol" aria-hidden="true">✓</span>FraudLens</a>
      <span className="service-status"><span className={`status-dot ${metadata ? 'ready' : ''}`} aria-hidden="true" />{metadata ? 'Ready to check' : loading ? 'Connecting…' : 'Unavailable'}</span>
    </div></header>
    <main>
      <section className="intro"><h1>Check a transaction</h1><p>Enter the payment details to see whether it needs a closer look.</p></section>
      {loading && !metadata && <p className="loading-message" role="status">Getting the workspace ready…</p>}
      {error && !metadata && <div className="error-banner" role="alert"><p>{error}</p><button type="button" onClick={() => setRetryCount((count) => count + 1)} disabled={loading}>Try again</button></div>}
      {metadata && <div className="workspace">
        <section className="input-panel" aria-labelledby="payment-heading">
          <div className="panel-intro"><h2 id="payment-heading">Payment details</h2><p>Start with an example, then change what you need.</p></div>
          <form onSubmit={handleSubmit} onInvalidCapture={revealInvalidField}>
            <fieldset disabled={busy} className="all-fields">
              <div className="sample-picker"><label htmlFor="example-transaction">Example transaction</label><select id="example-transaction" value={selectedSample} onChange={(event) => loadSample(event.target.value)}>
                {selectedSample === 'custom' && <option value="custom" disabled>Custom transaction</option>}
                {metadata.samples.map((sample) => <option key={sample.id} value={sample.id}>{SAMPLE_LABELS[sample.id] ?? sample.name}</option>)}
              </select></div>
              <div className="payment-fields fields-grid">{PAYMENT_FIELDS.map((field) => <Field key={field.name} field={field} value={form[field.name]} options={metadata.options} onChange={updateField} />)}</div>
              <details className="context-details">
                <summary>Account & security details <span className="optional-label">Optional to edit</span></summary>
                <p className="context-note">Already filled from the selected example’s account history. Update only if needed.</p>
                {CONTEXT_GROUPS.map((group) => <section className="context-group" key={group.title}><h3>{group.title}</h3><div className="fields-grid">{group.fields.map((field) => <Field key={field.name} field={field} value={form[field.name]} options={metadata.options} onChange={updateField} />)}</div></section>)}
              </details>
            </fieldset>
            <div className="submit-area">
              {error && <div className="error-banner" role="alert"><p>{error}</p></div>}
              <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Checking transaction…' : 'Check transaction'}<span aria-hidden="true">→</span></button>
              <span className="sr-only" role="status">{busy ? 'Check in progress' : result && !stale ? `Check complete: ${percent(result.fraud_probability)} chance of fraud. ${RISK_MESSAGES[result.risk_level].label}.` : ''}</span>
            </div>
          </form>
        </section>
        <Results result={result} stale={stale} busy={busy} />
      </div>}
    </main>
    <footer><span>Demo workspace · Example account data · No bank account connected</span></footer>
  </>;
}
