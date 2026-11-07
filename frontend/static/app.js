'use strict';

const form = document.getElementById('transaction-form');
const fields = document.getElementById('transaction-fields');
const exampleSelect = document.getElementById('example-transaction');
const checkButton = document.getElementById('check-button');
const retryButton = document.getElementById('retry-button');
const resultPanel = document.getElementById('result-panel');
const state = { metadata: null, result: null, busy: false, revision: 0 };

const numericFields = [
  'amount', 'card_present', 'device_known', 'ip_risk_score', 'is_foreign_txn',
  'time_since_last_s', 'velocity_1h', 'amount_vs_avg_ratio', 'account_age_days', 'has_2fa',
];
const sampleLabels = { original: 'Card payment', unfamiliar: 'Online purchase', unusual: 'Rapid payments' };
const paymentLabels = {
  atm: 'ATM', pos_terminal: 'Card terminal', web_browser: 'Online purchase',
  mobile_app: 'Mobile app', phone_ivr: 'Phone payment', gas_station: 'Fuel station', online_retail: 'Online shopping',
};
const checkLabels = {
  'Amount and deviation': 'Amount compared with usual spending',
  Velocity: 'Recent payment activity', 'IP risk': 'Connection security',
  Device: 'Device familiarity', Authentication: 'Extra verification',
  'Merchant geography': 'Payment location', 'Merchant category': 'Type of business',
  'Card channel': 'How the card was used', 'Account age': 'Account history', Time: 'Time of payment',
};
const riskMessages = {
  Low: { label: 'Low risk', title: 'No strong warning signs', action: 'Continue with your usual transaction checks.' },
  Review: { label: 'Needs review', title: 'Take a closer look', action: 'Review the payment details and confirm anything unusual with the customer.' },
  High: { label: 'High risk', title: 'Verify this payment', action: 'Verify the payment with the customer before proceeding.' },
};
const countryNames = new Intl.DisplayNames(['en'], { type: 'region' });

function humanize(value) {
  return paymentLabels[value] || value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function percent(value) {
  return `${(value * 100).toLocaleString('en-US', {
    minimumFractionDigits: 2, maximumFractionDigits: value < 0.001 ? 4 : 2,
  })}%`;
}

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => null);
    if (!response.ok || body === null) {
      throw new Error(response.status === 422
        ? 'Some payment or account details are missing or invalid. Please check them and try again.'
        : 'The transaction could not be checked. Please try again.');
    }
    return body;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) {
      throw new Error('The checking service is unavailable. Please try again shortly.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function showError(message, canRetry = false) {
  document.getElementById('error-message').textContent = message;
  document.getElementById('error-banner').hidden = false;
  retryButton.hidden = !canRetry;
}

function clearError() {
  document.getElementById('error-banner').hidden = true;
  retryButton.hidden = true;
}

function setBusy(busy) {
  state.busy = busy;
  fields.disabled = busy || !state.metadata;
  checkButton.disabled = busy || !state.metadata;
  retryButton.disabled = busy;
  document.getElementById('check-label').textContent = busy && state.metadata ? 'Checking transaction...' : 'Check transaction';
  resultPanel.setAttribute('aria-busy', String(busy));
}

function markChanged() {
  state.revision += 1;
  clearError();
  if (state.result) document.getElementById('stale-notice').hidden = false;
  document.getElementById('live-status').textContent = '';
}

function setOptions(select, values, labelFor) {
  const options = values.map((value) => new Option(labelFor(value), value));
  select.replaceChildren(...options);
}

function fillExample(id) {
  const sample = state.metadata.samples.find((item) => item.id === id);
  if (!sample) return;
  const values = { ...sample.transaction };
  const date = new Date(values.timestamp);
  values.timestamp = new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
  for (const [name, value] of Object.entries(values)) {
    const input = form.elements.namedItem(name);
    if (!input) continue;
    if (input.tagName === 'SELECT' && !Array.from(input.options).some((option) => option.value === String(value))) {
      input.add(new Option(name === 'merchant_country' ? countryNames.of(value) : humanize(String(value)), value));
    }
    input.value = value;
  }
  exampleSelect.querySelector('option[value="custom"]')?.remove();
  exampleSelect.value = id;
  markChanged();
}

async function loadWorkspace() {
  state.metadata = null;
  clearError();
  setBusy(true);
  document.getElementById('service-status').textContent = 'Connecting...';
  document.getElementById('status-dot').classList.remove('ready');
  document.getElementById('loading-message').hidden = false;
  try {
    const metadata = await requestJson('/api/metadata');
    if (!metadata.samples?.length || !metadata.options) throw new Error('The workspace could not be loaded. Please try again.');
    state.metadata = metadata;
    for (const name of ['merchant_category', 'merchant_country', 'device_type']) {
      setOptions(form.elements.namedItem(name), metadata.options[name],
        (value) => name === 'merchant_country' ? countryNames.of(value) : humanize(value));
    }
    exampleSelect.replaceChildren(...metadata.samples.map((sample) => new Option(sampleLabels[sample.id] || sample.name, sample.id)));
    fillExample(metadata.samples[0].id);
    document.getElementById('timestamp-help').textContent = `Your local time · ${Intl.DateTimeFormat().resolvedOptions().timeZone}`;
    document.getElementById('service-status').textContent = 'Ready to check';
    document.getElementById('status-dot').classList.add('ready');
  } catch (error) {
    state.metadata = null;
    document.getElementById('service-status').textContent = 'Unavailable';
    showError(error.message, true);
  } finally {
    document.getElementById('loading-message').hidden = true;
    setBusy(false);
  }
}

function readTransaction() {
  const transaction = Object.fromEntries(new FormData(form));
  for (const name of numericFields) {
    if (transaction[name] === '' || !Number.isFinite(Number(transaction[name]))) {
      throw new Error('Please complete the payment and account details.');
    }
    transaction[name] = Number(transaction[name]);
  }
  transaction.timestamp = new Date(transaction.timestamp).toISOString();
  transaction.merchant_country = transaction.merchant_country.toUpperCase();
  return transaction;
}

function renderChecks(explanation) {
  const checks = [...(explanation?.risk_increasing_characteristics || []), ...(explanation?.risk_decreasing_characteristics || [])]
    .filter((item) => checkLabels[item.characteristic_group])
    .sort((a, b) => Math.abs(b.calibrated_logodds_effect) - Math.abs(a.calibrated_logodds_effect))
    .slice(0, 3);
  const list = document.getElementById('check-list');
  list.replaceChildren(...checks.map((check) => {
    const item = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = checkLabels[check.characteristic_group];
    const direction = document.createElement('span');
    const higher = check.calibrated_logodds_effect > 0;
    direction.className = `check-direction ${higher ? 'higher' : 'lower'}`;
    direction.textContent = higher ? 'Raises concern' : 'Lowers concern';
    item.append(label, direction);
    return item;
  }));
  const details = document.getElementById('check-details');
  details.hidden = checks.length === 0;
  details.open = false;
}

function renderResult(result, submittedRevision) {
  const message = riskMessages[result.risk_level];
  const probability = result.fraud_probability;
  if (!message || !Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error('The result could not be read. Please try again.');
  }
  state.result = result;
  document.getElementById('empty-result').hidden = true;
  document.getElementById('assessment').hidden = false;
  resultPanel.classList.remove('empty-result');
  const riskClass = result.risk_level.toLowerCase();
  const badge = document.getElementById('risk-badge');
  badge.textContent = message.label;
  badge.className = `risk-badge ${riskClass}`;
  document.getElementById('fraud-probability').textContent = percent(probability);
  document.getElementById('probability-meter').setAttribute('aria-valuenow', probability * 100);
  const fill = document.getElementById('probability-fill');
  fill.style.width = `${probability * 100}%`;
  fill.className = riskClass;
  document.getElementById('next-step-title').textContent = message.title;
  document.getElementById('next-step-action').textContent = message.action;
  document.getElementById('stale-notice').hidden = state.revision === submittedRevision;
  renderChecks(result.explanation);
  document.getElementById('live-status').textContent = `Check complete: ${percent(probability)} chance of fraud. ${message.label}.`;
  const heading = document.getElementById('result-heading');
  heading.focus({ preventScroll: true });
  heading.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
}

async function submitTransaction(event) {
  event.preventDefault();
  if (state.busy || !state.metadata || !form.reportValidity()) return;
  clearError();
  try {
    const transaction = readTransaction();
    const revision = state.revision;
    setBusy(true);
    const result = await requestJson('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(transaction),
    });
    renderResult(result, revision);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function onFieldEdited(event) {
  if (!event.target.name || state.busy) return;
  markChanged();
  if (!exampleSelect.querySelector('option[value="custom"]')) {
    const custom = new Option('Custom transaction', 'custom');
    custom.disabled = true;
    exampleSelect.prepend(custom);
  }
  exampleSelect.value = 'custom';
}

form.addEventListener('submit', submitTransaction);
form.addEventListener('input', (event) => { if (event.target.tagName !== 'SELECT') onFieldEdited(event); });
form.addEventListener('change', (event) => { if (event.target.tagName === 'SELECT') onFieldEdited(event); });
form.addEventListener('invalid', (event) => {
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
exampleSelect.addEventListener('change', () => { if (!state.busy) fillExample(exampleSelect.value); });
retryButton.addEventListener('click', loadWorkspace);
loadWorkspace();
