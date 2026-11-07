// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App.jsx';
import transaction from '../../example_frontend_transaction.json';
import prediction from '../../example_frontend_prediction.json';

const unusual = { ...transaction, amount: 4800, ip_risk_score: 97, velocity_1h: 12, has_2fa: 0 };
const metadata = {
  model_name: 'XGBoost full',
  thresholds: { review: 0.0224948615, high: 0.8523615414 },
  options: { merchant_category: ['travel', 'grocery'], merchant_country: ['US', 'GB'], device_type: ['pos_terminal', 'web_browser'] },
  samples: [
    { id: 'original', name: 'Notebook example', transaction },
    { id: 'unusual', name: 'Rapid payments', transaction: unusual },
  ],
};

function mockApi(result = prediction) {
  const fetchMock = vi.fn(async (path) => ({ ok: true, json: async () => path === '/api/metadata' ? metadata : result }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('employee transaction check', () => {
  it('shows six payment fields and keeps prefilled account context collapsed', async () => {
    mockApi();
    render(<App />);
    await screen.findByRole('heading', { name: 'Payment details' });
    expect(document.querySelectorAll('.payment-fields input, .payment-fields select').length).toBe(6);
    expect(document.querySelector('.context-details').open).toBe(false);
    expect(screen.getByLabelText('Amount (USD)').value).toBe('1195.62');
    expect(screen.getByLabelText('Example transaction').value).toBe('original');
    expect(screen.getByRole('option', { name: 'United States' })).toBeTruthy();
    expect(screen.getByRole('option', { name: 'Card terminal' })).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/XGBoost|FastAPI|TreeSHAP|log-odds|API response|risk thresholds|Notebook/i);
  });

  it('sends all required context with edited payment details and preserves the local date instant', async () => {
    const fetchMock = mockApi();
    render(<App />);
    const amount = await screen.findByLabelText('Amount (USD)');
    fireEvent.change(amount, { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    await screen.findByText('Low risk');
    const predictionCall = fetchMock.mock.calls.find(([path]) => path === '/api/predict');
    const body = JSON.parse(predictionCall[1].body);
    expect(body.amount).toBe(0);
    expect(new Date(body.timestamp).getTime()).toBe(new Date(transaction.timestamp).getTime());
    expect(body.timestamp).toMatch(/Z$/);
    expect(body.device_known).toBe(1);
    expect(body.ip_risk_score).toBe(transaction.ip_risk_score);
    expect(Object.keys(body)).toHaveLength(14);
    expect(screen.getByText('0.0038%')).toBeTruthy();
    expect(document.querySelector('.check-details').open).toBe(false);
    expect(document.querySelectorAll('.check-details li').length).toBeLessThanOrEqual(3);
    expect(document.body.textContent).not.toMatch(/XGBoost|FastAPI|TreeSHAP|log-odds|API response|risk thresholds|card_not_present|conditional probabilities/i);
    expect(screen.queryByRole('table')).toBeNull();
  });

  it.each([
    ['Low', 'Low risk', 'No strong warning signs', 0.001],
    ['Review', 'Needs review', 'Take a closer look', 0.38],
    ['High', 'High risk', 'Verify this payment', 0.99],
  ])('gives an understandable next step for %s risk', async (risk, label, title, probability) => {
    mockApi({ ...prediction, risk_level: risk, fraud_probability: probability });
    render(<App />);
    await screen.findByLabelText('Amount (USD)');
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    expect(await screen.findByText(label)).toBeTruthy();
    expect(screen.getByRole('heading', { name: title })).toBeTruthy();
  });

  it('loads another example and marks a previous assessment outdated', async () => {
    mockApi();
    render(<App />);
    await screen.findByLabelText('Amount (USD)');
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    await screen.findByText('Low risk');
    fireEvent.change(screen.getByLabelText('Example transaction'), { target: { value: 'unusual' } });
    expect(screen.getByLabelText('Amount (USD)').value).toBe('4800');
    expect(screen.getByLabelText('Connection risk (0–100)').value).toBe('97');
    expect(screen.getByText(/Details changed/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    await waitFor(() => expect(screen.queryByText(/Details changed/)).toBeNull());
  });

  it('reveals invalid account details even after their section was closed', async () => {
    const fetchMock = mockApi();
    render(<App />);
    const connection = await screen.findByLabelText('Connection risk (0–100)');
    fireEvent.change(connection, { target: { value: '-1' } });
    const details = document.querySelector('.context-details');
    details.open = false;
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    expect(details.open).toBe(true);
    expect(fetchMock.mock.calls.some(([path]) => path === '/api/predict')).toBe(false);
  });

  it('allows retrying a connection failure without exposing setup instructions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValueOnce(new TypeError('network error'))
      .mockResolvedValue({ ok: true, json: async () => metadata }));
    render(<App />);
    expect((await screen.findByRole('alert')).textContent).toContain('checking service is unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { name: 'Payment details' })).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/FastAPI|port 8000|\bAPI\b/i);
  });

  it.each([422, 500])('shows a plain failure message for HTTP %s without internal diagnostics', async (status) => {
    vi.stubGlobal('fetch', vi.fn(async (path) => path === '/api/metadata'
      ? { ok: true, json: async () => metadata }
      : { ok: false, status, json: async () => ({ detail: 'XGBoost internal error in model inference' }) }));
    render(<App />);
    await screen.findByLabelText('Amount (USD)');
    fireEvent.click(screen.getByRole('button', { name: 'Check transaction' }));
    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.queryByText('Low risk')).toBeNull();
    expect(document.body.textContent).not.toMatch(/XGBoost|model inference|HTTP/);
  });
});
