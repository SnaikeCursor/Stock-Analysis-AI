/**
 * Typed fetch wrapper for the Stock Analysis AI FastAPI backend (`/api/*`).
 * Set `VITE_API_BASE_URL` (e.g. `http://localhost:8000`) in `.env` for local dev.
 */

function normalizeApiBase(raw: string | undefined): string {
  const s = raw?.trim().replace(/\/$/, '')
  if (!s) return 'http://localhost:8000'
  if (/^https?:\/\//.test(s)) return s
  return `https://${s}`
}

const API_BASE = normalizeApiBase(import.meta.env.VITE_API_BASE_URL as string | undefined)

/** localStorage key for anonymous portfolio identity (UUID v4). */
export const PORTFOLIO_USER_ID_KEY = 'portfolio_user_id'

let _portfolioUserIdMemory: string | null = null

/** Return or create a persistent UUID for `X-User-ID` (manual portfolio). */
export function getPortfolioUserId(): string {
  try {
    let id = localStorage.getItem(PORTFOLIO_USER_ID_KEY)
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem(PORTFOLIO_USER_ID_KEY, id)
    }
    return id
  } catch {
    if (!_portfolioUserIdMemory) _portfolioUserIdMemory = crypto.randomUUID()
    return _portfolioUserIdMemory
  }
}

/** Set portfolio identity from a full UUID string (e.g. another device). Returns false if invalid. */
export function setPortfolioUserId(uuid: string): boolean {
  const s = uuid.trim()
  const re =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
  if (!re.test(s)) return false
  try {
    localStorage.setItem(PORTFOLIO_USER_ID_KEY, s.toLowerCase())
    return true
  } catch {
    return false
  }
}

/** Short display code (first segment of UUID, 8 hex chars). */
export function getPortfolioDisplayCode(): string {
  const id = getPortfolioUserId()
  return id.split('-')[0] ?? id.slice(0, 8)
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text()
  if (!text) return undefined as T
  return JSON.parse(text) as T
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...init?.headers,
      'X-User-ID': getPortfolioUserId(),
    },
  })

  if (!res.ok) {
    const body = await res.text()
    throw new ApiError(res.status, body || res.statusText)
  }

  if (res.status === 204) return undefined as T

  const contentType = res.headers.get('content-type')
  if (contentType?.includes('application/json')) {
    return parseJson<T>(res)
  }

  return undefined as T
}

function withJson(init?: RequestInit): RequestInit {
  return {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  }
}

// --- Types (aligned with backend Pydantic / dict responses) ---

export type HealthResponse = {
  status: string
  data_loaded: boolean
  model_loaded: boolean
}

export type DashboardSignal = {
  id: number
  cutoff_date: string
  created_at: string | null
  status: string
  /** From persisted signal JSON; omitted for legacy rows. */
  n_positions?: number
  requested_top_n?: number | null
}

export type DashboardPosition = {
  id: number
  ticker: string
  weight: number
  entry_price: number | null
  entry_date: string | null
  current_price: number | null
  pnl_pct: number | null
}

export type DashboardAlert = {
  id: number
  type: string
  message: string
  created_at: string | null
}

export type ModelInfo = {
  phase: string
  rebalance_freq: string
  description: string
}

export type DashboardResponse = {
  signal: DashboardSignal | null
  positions: DashboardPosition[]
  next_rebalancing: { date: string; days_until: number }
  alerts: DashboardAlert[]
  model_info?: ModelInfo | null
}

export type SignalPosition = {
  ticker: string
  weight: number
  predicted_return: number
  current_price?: number | null
}

export type SignalOut = {
  id: number
  cutoff_date: string
  portfolio: SignalPosition[]
  status: string
  created_at: string | null
  model_phase?: string
  rebalance_freq?: string
  /** Requested long count; may exceed len(portfolio) if coverage is thin. */
  requested_top_n?: number | null
}

export type GenerateSignalBody = {
  cutoff_date?: string | null
  top_n?: number
  max_weight?: number
}

export type YearMetrics = {
  cumulative_return: number
  annualized_return: number
  volatility: number
  sharpe_ratio: number
  max_drawdown: number
  n_trading_days: number
}

export type YearPerformance = {
  long_only: YearMetrics
  benchmark: YearMetrics
  costs_bps: number
}

export type HistoryPerformanceResponse = {
  per_year: Record<string, YearPerformance>
  total_costs_bps: number
  rebalance_freq: number
}

export type QuarterlyDetail = {
  year: number
  quarter: string
  cutoff: string
  start: string
  end: string
  n_positions: number
  n_swapped: number
  turnover_pct: number
  cost_bps: number
  cum_return: number
  sharpe: number
  max_dd: number
  n_trading_days: number
}

export type HistoryQuarterlyResponse = {
  quarterly_detail: QuarterlyDetail[]
  total_costs_bps: number
}

export type SimulateTimelinePoint = {
  date: string
  portfolio_value: number
  benchmark_value: number
}

export type SimulateTransaction = {
  date: string
  ticker: string
  action: 'buy' | 'sell'
  shares: number
  price: number
  value: number
}

export type SimulateSummary = {
  initial_capital: number
  final_value: number
  total_return: number
  annualized_return: number
  sharpe_ratio: number
  max_drawdown: number
  total_costs: number
  n_trades: number
  benchmark_final_value: number
  benchmark_total_return: number
}

export type SimulateResponse = {
  timeline: SimulateTimelinePoint[]
  transactions: SimulateTransaction[]
  summary: SimulateSummary
}

export type PositionDetail = {
  id: number
  ticker: string
  weight: number
  shares?: number | null
  entry_price: number | null
  entry_date: string | null
  entry_total?: number | null
  exit_price: number | null
  exit_date: string | null
  exit_total?: number | null
  pnl_pct: number | null
}

export type PnlEntry = {
  id: number
  ticker: string
  weight: number
  shares: number | null
  entry_price: number | null
  entry_date: string | null
  entry_total: number | null
  current_price: number | null
  current_value: number | null
  pnl_pct: number | null
  pnl_abs: number | null
}

/** Same row shape for dashboard positions and live P&L (interchangeable in tables). */
export type LivePositionRow = DashboardPosition

export type AlertOut = {
  id: number
  type: string
  message: string
  created_at: string | null
  read: boolean
}

export type ClosePositionBody = {
  exit_price: number
  exit_date?: string | null
}

/** Partial update for `PUT /api/portfolio/positions/{id}` — aligns with backend `UpdatePositionRequest`. */
export type UpdatePositionBody = {
  shares?: number | null
  entry_price?: number | null
  entry_date?: string | null
  exit_price?: number | null
  exit_date?: string | null
}

export type ActivatePortfolioPositionInput = {
  ticker: string
  shares: number
  entry_price: number
  entry_date: string
}

/** `POST /api/portfolio/activate` — activate a PENDING signal with share counts and entry data. */
export type ActivatePortfolioBody = {
  signal_id: number
  investment_amount: number
  positions: ActivatePortfolioPositionInput[]
}

export type ActivatePortfolioResponse = {
  signal_id: number
  status: string
  positions_activated: number
  investment_amount: number
}

export type RebalanceProposalStatus = 'pending' | 'executed' | 'dismissed'

/** One row inside `instructions` from `compute_rebalance_instructions` (backend JSON). */
export type RebalanceInstructionRow = {
  ticker: string
  action: 'buy' | 'sell' | 'hold' | 'skip' | string
  shares: number
  estimated_price?: number
  estimated_value?: number
  reason?: string
}

export type RebalanceInstructionsPayload = {
  portfolio_value: number
  swap_count: number
  instructions: RebalanceInstructionRow[]
  note?: string
}

/** `GET /api/portfolio/rebalance-proposal` — pending proposal, or null if none. */
export type RebalanceProposal = {
  id: number
  old_signal_id: number
  new_signal_id: number
  created_at: string | null
  status: RebalanceProposalStatus
  portfolio_value: number | null
  swap_count: number | null
  instructions: RebalanceInstructionRow[]
  note: string | null
}

export type ExecutedTrade = {
  ticker: string
  action: string
  shares: number
  price: number
  date: string
}

/** `POST /api/portfolio/execute-rebalance` — confirm executed swaps. */
export type ExecuteRebalanceBody = {
  rebalance_id: number
  executed_trades: ExecutedTrade[]
}

export type ExecuteRebalanceResponse = {
  rebalance_id: number
  status: RebalanceProposalStatus
  signal: SignalOut
  positions: PositionDetail[]
}

export type PortfolioHistorySignal = {
  signal_id: number
  cutoff_date: string
  regime_label?: string | null
  status: string
  created_at: string | null
  positions: PositionDetail[]
}

/** Manual user portfolio (`/api/me/portfolio`) — open position row with live marks. */
export type MyOpenPosition = {
  id: number
  ticker: string
  shares: number
  entry_price: number
  entry_date: string
  entry_total: number
  entry_fee: number
  current_price: number | null
  current_value: number | null
  pnl_pct: number | null
  pnl_abs: number | null
  status: string
}

export type MyClosedPosition = {
  id: number
  ticker: string
  shares: number
  entry_price: number
  entry_date: string
  entry_total: number
  entry_fee: number
  exit_price: number | null
  exit_date: string | null
  exit_total: number | null
  exit_fee: number | null
  pnl_abs: number | null
  pnl_pct: number | null
  status: string
}

export type MyPortfolioOverview = {
  user_uuid: string
  cash_balance: number
  open_positions: MyOpenPosition[]
  closed_positions: MyClosedPosition[]
}

export type MyPortfolioSummary = {
  cash_balance: number
  total_portfolio_value: number
  unrealized_pnl: number
  realized_pnl: number
  total_fees_paid: number
  open_notional_at_cost: number
  open_market_value: number
  n_open: number
  n_closed: number
}

export type SwissquoteFeeEstimate = {
  volume_chf: number
  fee_chf: number
}

export type ApplySignalResponse = {
  signal_id: number
  positions_created: {
    ticker: string
    shares: number
    entry_price: number
    entry_total: number
    fee: number
    weight: number
  }[]
  total_invested: number
  cash_remaining: number
}

export type PortfolioPerformance = {
  total_deposited: number
  total_withdrawn: number
  current_value: number
  open_market_value: number
  cash_balance: number
  net_invested: number
  total_pnl_abs: number
  total_return_pct: number | null
}

// --- API ---

export const api = {
  getHealth: () => request<HealthResponse>('/api/health'),

  getDashboard: () => request<DashboardResponse>('/api/dashboard'),

  generateSignal: (body: GenerateSignalBody = {}) =>
    request<SignalOut>('/api/signals/generate', withJson({ method: 'POST', body: JSON.stringify(body) })),

  getLatestSignal: () => request<SignalOut | null>('/api/signals/latest'),

  getSignalHistory: () => request<SignalOut[]>('/api/signals/history'),

  getPerformance: (startYear = 2015, endYear = 2025) =>
    request<HistoryPerformanceResponse>(
      `/api/history/performance?${new URLSearchParams({
        start_year: String(startYear),
        end_year: String(endYear),
      })}`,
    ),

  getQuarterly: (startYear = 2015, endYear = 2025) =>
    request<HistoryQuarterlyResponse>(
      `/api/history/quarterly?${new URLSearchParams({
        start_year: String(startYear),
        end_year: String(endYear),
      })}`,
    ),

  simulateBacktest: (body: { start_date: string; initial_capital: number; costs_bps: number }) =>
    request<SimulateResponse>(
      '/api/history/simulate',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  getPortfolio: () => request<PositionDetail[]>('/api/portfolio'),

  getPortfolioPnl: () => request<PnlEntry[]>('/api/portfolio/pnl'),

  getPortfolioHistory: () => request<PortfolioHistorySignal[]>('/api/portfolio/history'),

  closePosition: (positionId: number, body: ClosePositionBody) =>
    request<PositionDetail>(
      `/api/portfolio/positions/${positionId}/close`,
      withJson({ method: 'PUT', body: JSON.stringify(body) }),
    ),

  updatePosition: (positionId: number, body: UpdatePositionBody) =>
    request<PositionDetail>(
      `/api/portfolio/positions/${positionId}`,
      withJson({ method: 'PUT', body: JSON.stringify(body) }),
    ),

  activatePortfolio: (body: ActivatePortfolioBody) =>
    request<ActivatePortfolioResponse>(
      '/api/portfolio/activate',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  getRebalanceProposal: () =>
    request<RebalanceProposal | null>('/api/portfolio/rebalance-proposal'),

  executeRebalance: (body: ExecuteRebalanceBody) =>
    request<ExecuteRebalanceResponse>(
      '/api/portfolio/execute-rebalance',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  dismissRebalance: (rebalanceId: number) =>
    request<{ id: number; status: string; message: string }>(
      '/api/portfolio/dismiss-rebalance',
      withJson({ method: 'POST', body: JSON.stringify({ rebalance_id: rebalanceId }) }),
    ),

  getAlerts: (unreadOnly = false) =>
    request<AlertOut[]>(`/api/alerts?${new URLSearchParams({ unread_only: String(unreadOnly) })}`),

  markAlertRead: (alertId: number) =>
    request<AlertOut>(`/api/alerts/${alertId}/read`, { method: 'PUT' }),

  /** Anonymous manual portfolio (requires `X-User-ID` — sent automatically). */
  getMyPortfolio: () => request<MyPortfolioOverview>('/api/me/portfolio'),

  getMyPortfolioSummary: () => request<MyPortfolioSummary>('/api/me/portfolio/summary'),

  depositMyPortfolio: (body: { amount: number }) =>
    request<{ cash_balance: number }>(
      '/api/me/portfolio/deposit',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  addMyPosition: (body: { ticker: string; shares: number; entry_price: number; entry_date: string }) =>
    request<Record<string, unknown>>(
      '/api/me/portfolio/positions',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  closeMyPosition: (positionId: number, body: { exit_price: number; exit_date?: string | null }) =>
    request<Record<string, unknown>>(
      `/api/me/portfolio/positions/${positionId}/close`,
      withJson({ method: 'PUT', body: JSON.stringify(body) }),
    ),

  deleteMyPosition: (positionId: number) =>
    request<void>(`/api/me/portfolio/positions/${positionId}`, { method: 'DELETE' }),

  getSwissquoteFeeEstimate: (volumeChf: number) =>
    request<SwissquoteFeeEstimate>(
      `/api/me/swissquote-fee?${new URLSearchParams({ volume_chf: String(volumeChf) })}`,
    ),

  applySignal: (body: { signal_id: number; investment_amount: number }) =>
    request<ApplySignalResponse>(
      '/api/me/portfolio/apply-signal',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  withdrawMyPortfolio: (body: { amount: number }) =>
    request<{ cash_balance: number }>(
      '/api/me/portfolio/withdraw',
      withJson({ method: 'POST', body: JSON.stringify(body) }),
    ),

  getMyPortfolioPerformance: () =>
    request<PortfolioPerformance>('/api/me/portfolio/performance'),
}

export function getApiBaseUrl(): string {
  return API_BASE
}
