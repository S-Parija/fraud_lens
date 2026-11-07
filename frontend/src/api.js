/** Fetch JSON and turn FastAPI validation errors into readable form messages. */
export async function requestJson(path, options = {}) {
  let response;
  try {
    const timeout = AbortSignal.timeout(30000);
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
    response = await fetch(path, { ...options, signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('The checking service is unavailable. Please try again shortly.');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = response.status === 422
      ? 'Some payment or account details are missing or invalid. Please check them and try again.'
      : 'The transaction could not be checked. Please try again.';
    throw new Error(message);
  }
  return body;
}
