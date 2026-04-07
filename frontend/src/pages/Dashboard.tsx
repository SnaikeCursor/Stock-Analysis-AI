import { Fragment, useCallback, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { format } from 'date-fns'
import {
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRightLeft,
  ArrowUp,
  CalendarClock,
  Check,
  Info,
  Loader2,
  RefreshCw,
  Sparkles,
  TrendingUp,
  Wallet,
  X,
  Zap,
} from 'lucide-react'

import {
  api,
  type ActivatePortfolioBody,
  type ActivatePortfolioResponse,
  type ExecutedTrade,
  type PnlEntry,
  type RebalanceInstructionRow,
  type RebalanceProposal,
  type SignalOut,
} from '@/lib/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { PortfolioTable } from '@/components/PortfolioTable'

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

function fmtPrice(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtChf(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

function fmtSizingChf(v: number) {
  return v.toLocaleString('de-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtSizingPct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return format(new Date(iso), 'dd.MM.yyyy')
  } catch {
    return iso
  }
}

function todayIso() {
  return new Date().toISOString().slice(0, 10)
}

// ---------------------------------------------------------------------------
// KPI computation (amount-based)
// ---------------------------------------------------------------------------

function computePortfolioKpis(positions: PnlEntry[]) {
  let totalInvested = 0
  let totalCurrentValue = 0
  let positionsWithValue = 0

  for (const p of positions) {
    if (p.entry_total != null) totalInvested += p.entry_total
    if (p.current_value != null) {
      totalCurrentValue += p.current_value
      positionsWithValue++
    }
  }

  const totalPnlAbs = totalCurrentValue - totalInvested
  const totalPnlPct = totalInvested > 0 ? totalPnlAbs / totalInvested : null

  const withPnl = positions.filter((p) => p.pnl_pct != null)
  const sorted = [...withPnl].sort((a, b) => (b.pnl_pct ?? 0) - (a.pnl_pct ?? 0))
  const best = sorted[0] ?? null
  const worst = sorted[sorted.length - 1] ?? null

  return {
    totalInvested,
    totalCurrentValue,
    totalPnlAbs,
    totalPnlPct,
    count: positions.length,
    positionsWithValue,
    best,
    worst,
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const inputClass =
  'flex h-8 w-full min-w-0 rounded-md border border-input bg-transparent px-2 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'

const INPUT_CLS =
  'flex h-7 w-full rounded border border-border bg-background px-2 text-sm tabular-nums outline-none focus:ring-2 focus:ring-ring/50'

// ---------------------------------------------------------------------------
// Small utility components
// ---------------------------------------------------------------------------

function KpiCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
}: {
  title: string
  value: string
  subtitle?: string
  icon: React.ElementType
  trend?: 'up' | 'down' | 'neutral'
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardDescription className="flex items-center gap-1.5">
          <Icon className="size-3.5 opacity-70" />
          {title}
        </CardDescription>
        <CardTitle
          className={cn(
            'text-xl tabular-nums',
            trend === 'up' && 'text-emerald-600',
            trend === 'down' && 'text-red-600',
          )}
        >
          <span className="inline-flex items-center gap-1">
            {trend === 'up' && <ArrowUp className="size-4" />}
            {trend === 'down' && <ArrowDown className="size-4" />}
            {value}
          </span>
        </CardTitle>
      </CardHeader>
      {subtitle && (
        <CardContent>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </CardContent>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Rebalancing Banner
// ---------------------------------------------------------------------------

type TradeEdit = {
  ticker: string
  action: string
  shares: string
  price: string
  date: string
}

function RebalancingBanner({
  proposal,
  onExecute,
  onDismiss,
  isExecuting,
  isDismissing,
}: {
  proposal: RebalanceProposal
  onExecute: (trades: ExecutedTrade[]) => void
  onDismiss: () => void
  isExecuting: boolean
  isDismissing: boolean
}) {
  const rows: RebalanceInstructionRow[] = proposal.instructions ?? []
  const swapCount = proposal.swap_count ?? 0
  const portfolioValue = proposal.portfolio_value ?? 0

  const [expanded, setExpanded] = useState(false)
  const [trades, setTrades] = useState<TradeEdit[]>(() =>
    rows
      .filter((r) => r.action !== 'skip')
      .map((r) => ({
        ticker: r.ticker,
        action: r.action,
        shares: String(r.shares),
        price: r.estimated_price != null ? String(r.estimated_price) : '',
        date: new Date().toISOString().slice(0, 10),
      })),
  )

  const updateTrade = (idx: number, field: keyof TradeEdit, value: string) => {
    setTrades((prev) => prev.map((t, i) => (i === idx ? { ...t, [field]: value } : t)))
  }

  const handleConfirm = () => {
    const executed: ExecutedTrade[] = trades
      .filter((t) => t.action !== 'hold' || parseInt(t.shares, 10) > 0)
      .map((t) => ({
        ticker: t.ticker,
        action: t.action,
        shares: parseInt(t.shares, 10) || 0,
        price: parseFloat(t.price.replace(',', '.')) || 0,
        date: t.date || new Date().toISOString().slice(0, 10),
      }))
    onExecute(executed)
  }

  const actionBadge = (action: string) => {
    const map: Record<string, { label: string; variant: 'default' | 'secondary' | 'outline' | 'destructive' }> = {
      buy: { label: 'Buy', variant: 'default' },
      sell: { label: 'Sell', variant: 'destructive' },
      hold: { label: 'Hold', variant: 'secondary' },
    }
    const cfg = map[action] ?? { label: action, variant: 'outline' as const }
    return (
      <Badge variant={cfg.variant} className="text-[0.65rem]">
        {cfg.label}
      </Badge>
    )
  }

  return (
    <Card className="border-amber-300 bg-amber-50/50 dark:border-amber-700 dark:bg-amber-950/20">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ArrowRightLeft className="size-4 text-amber-600" />
            Rebalancing faellig
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="tabular-nums text-xs">
              {swapCount} Swap{swapCount !== 1 ? 's' : ''}
            </Badge>
            <Badge variant="secondary" className="tabular-nums text-xs">
              Wert ~{fmtChf(portfolioValue)} CHF
            </Badge>
          </div>
        </div>
        <CardDescription>
          Neues Signal #{proposal.new_signal_id} ersetzt #{proposal.old_signal_id}
          {proposal.created_at && <> — erstellt {fmtDate(proposal.created_at)}</>}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {proposal.note && (
          <p className="text-sm text-muted-foreground">{proposal.note}</p>
        )}

        {!expanded ? (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2 text-sm">
              {rows
                .filter((r) => r.action === 'buy' || r.action === 'sell')
                .map((r) => (
                  <span key={r.ticker} className="inline-flex items-center gap-1.5">
                    {actionBadge(r.action)}
                    <span className="font-mono text-xs">
                      {r.shares}x {r.ticker}
                    </span>
                    {r.estimated_price != null && (
                      <span className="text-muted-foreground">~{fmtPrice(r.estimated_price)}</span>
                    )}
                  </span>
                ))}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={() => setExpanded(true)}>
                Details & bestaetigen
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground"
                onClick={onDismiss}
                disabled={isDismissing}
              >
                {isDismissing ? 'Verwerfen…' : 'Verwerfen'}
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Aktion</TableHead>
                  <TableHead>Ticker</TableHead>
                  <TableHead className="text-right">Stueck</TableHead>
                  <TableHead className="text-right">Preis</TableHead>
                  <TableHead className="text-right">Datum</TableHead>
                  <TableHead className="text-right">Wert</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.map((t, idx) => {
                  const shares = parseInt(t.shares, 10) || 0
                  const price = parseFloat(t.price.replace(',', '.')) || 0
                  const value = shares * price
                  return (
                    <TableRow key={t.ticker}>
                      <TableCell>{actionBadge(t.action)}</TableCell>
                      <TableCell className="font-mono text-sm font-medium">{t.ticker}</TableCell>
                      <TableCell className="text-right">
                        <input
                          type="number"
                          min={0}
                          className={cn(inputClass, 'w-16 text-right')}
                          value={t.shares}
                          onChange={(e) => updateTrade(idx, 'shares', e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <input
                          type="text"
                          inputMode="decimal"
                          className={cn(inputClass, 'w-24 text-right')}
                          value={t.price}
                          onChange={(e) => updateTrade(idx, 'price', e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <input
                          type="date"
                          className={cn(inputClass, 'w-32')}
                          value={t.date}
                          onChange={(e) => updateTrade(idx, 'date', e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {fmtChf(value)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" onClick={handleConfirm} disabled={isExecuting}>
                <Check className="size-3.5" />
                {isExecuting ? 'Ausfuehren…' : 'Swaps bestaetigen'}
              </Button>
              <Button size="sm" variant="outline" onClick={() => setExpanded(false)}>
                Zurueck
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground"
                onClick={onDismiss}
                disabled={isDismissing}
              >
                <X className="size-3.5" />
                {isDismissing ? 'Verwerfen…' : 'Verwerfen'}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Signal Generation — 2-step wizard (Generate → Configure Trades)
// ---------------------------------------------------------------------------

type SignalPhase = 'idle' | 'confirming' | 'sizing' | 'result'

type EditableRow = {
  ticker: string
  weight: number
  predicted_return: number
  currentPrice: number
  shares: number
  entryPrice: number
  entryDate: string
}

function StepIndicator({ step }: { step: 1 | 2 }) {
  const steps = [
    { n: 1, label: 'Generate' },
    { n: 2, label: 'Configure Trades' },
  ] as const

  return (
    <div className="flex items-center gap-2 text-xs">
      {steps.map((s, i) => (
        <Fragment key={s.n}>
          {i > 0 && <div className="h-px w-6 bg-border" />}
          <div
            className={cn(
              'flex items-center gap-1.5 rounded-full px-3 py-1 font-medium transition-colors',
              s.n === step
                ? 'bg-primary text-primary-foreground'
                : s.n < step
                  ? 'bg-primary/10 text-primary'
                  : 'bg-muted text-muted-foreground',
            )}
          >
            {s.n < step ? <Check className="size-3" /> : <span>{s.n}</span>}
            <span>{s.label}</span>
          </div>
        </Fragment>
      ))}
    </div>
  )
}

function SizingCard({
  signal,
  onBack,
  onSuccess,
}: {
  signal: SignalOut
  onBack: () => void
  onSuccess: (result: ActivatePortfolioResponse) => void
}) {
  const today = todayIso()
  const [investmentRaw, setInvestmentRaw] = useState('')
  const [rows, setRows] = useState<EditableRow[]>(() =>
    signal.portfolio.map((p) => ({
      ticker: p.ticker,
      weight: p.weight,
      predicted_return: p.predicted_return,
      currentPrice: p.current_price ?? 0,
      shares: 0,
      entryPrice: p.current_price ?? 0,
      entryDate: today,
    })),
  )

  const investment = parseFloat(investmentRaw) || 0

  const updateRow = (idx: number, patch: Partial<EditableRow>) =>
    setRows((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)))

  const handleInvestmentChange = (raw: string) => {
    setInvestmentRaw(raw)
    const amount = parseFloat(raw) || 0
    if (amount > 0) {
      setRows((prev) =>
        prev.map((r) => ({
          ...r,
          shares:
            r.entryPrice > 0
              ? Math.floor((amount * r.weight) / r.entryPrice)
              : 0,
        })),
      )
    }
  }

  const totalInvested = rows.reduce(
    (sum, r) => sum + r.shares * r.entryPrice,
    0,
  )
  const remainder = investment - totalInvested

  const activateMutation = useMutation({
    mutationFn: api.activatePortfolio,
    onSuccess: (data) => onSuccess(data),
  })

  const handleConfirm = () => {
    const body: ActivatePortfolioBody = {
      signal_id: signal.id,
      investment_amount: investment,
      positions: rows.map((r) => ({
        ticker: r.ticker,
        shares: r.shares,
        entry_price: r.entryPrice,
        entry_date: r.entryDate,
      })),
    }
    activateMutation.mutate(body)
  }

  const canConfirm =
    investment > 0 &&
    rows.every((r) => r.shares > 0 && r.entryPrice > 0) &&
    !activateMutation.isPending

  return (
    <Card className="border-primary/20">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="size-4 text-primary" />
          Configure Trades
        </CardTitle>
        <CardDescription>
          Signal #{signal.id} · {signal.cutoff_date}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="space-y-1.5">
          <label
            className="text-xs font-medium text-muted-foreground"
            htmlFor="investment"
          >
            Investment Amount (CHF)
          </label>
          <input
            id="investment"
            type="number"
            min={0}
            step={100}
            placeholder="e.g. 10000"
            value={investmentRaw}
            onChange={(e) => handleInvestmentChange(e.target.value)}
            className="flex h-9 w-full max-w-xs rounded-md border border-border bg-background px-3 text-sm tabular-nums outline-none focus:ring-2 focus:ring-ring/50"
          />
        </div>

        <div className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ticker</TableHead>
                <TableHead className="text-right">Weight</TableHead>
                <TableHead className="text-right">Mkt Price</TableHead>
                <TableHead className="w-20 text-right">Shares</TableHead>
                <TableHead className="w-24 text-right">Entry Price</TableHead>
                <TableHead className="w-32">Entry Date</TableHead>
                <TableHead className="text-right">Amount</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => {
                const amount = r.shares * r.entryPrice
                return (
                  <TableRow key={r.ticker}>
                    <TableCell className="font-medium">{r.ticker}</TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {fmtSizingPct(r.weight)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {r.currentPrice > 0 ? fmtSizingChf(r.currentPrice) : '—'}
                    </TableCell>
                    <TableCell className="text-right">
                      <input
                        type="number"
                        min={0}
                        step={1}
                        className={cn(INPUT_CLS, 'w-20 text-right')}
                        value={r.shares > 0 ? r.shares : ''}
                        onChange={(e) =>
                          updateRow(i, {
                            shares: Math.max(0, parseInt(e.target.value) || 0),
                          })
                        }
                      />
                    </TableCell>
                    <TableCell className="text-right">
                      <input
                        type="number"
                        min={0}
                        step={0.01}
                        className={cn(INPUT_CLS, 'w-24 text-right')}
                        value={r.entryPrice > 0 ? r.entryPrice : ''}
                        onChange={(e) =>
                          updateRow(i, {
                            entryPrice: Math.max(0, parseFloat(e.target.value) || 0),
                          })
                        }
                      />
                    </TableCell>
                    <TableCell>
                      <input
                        type="date"
                        className={cn(INPUT_CLS, 'w-32')}
                        value={r.entryDate}
                        onChange={(e) =>
                          updateRow(i, { entryDate: e.target.value })
                        }
                      />
                    </TableCell>
                    <TableCell className="text-right tabular-nums font-medium">
                      {amount > 0 ? fmtSizingChf(amount) : '—'}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>

        {investment > 0 && (
          <div className="flex items-center gap-4 text-sm">
            <span className="text-muted-foreground">
              Total:{' '}
              <span className="font-medium text-foreground">
                {fmtSizingChf(totalInvested)} CHF
              </span>
            </span>
            <Separator orientation="vertical" className="h-4" />
            <span className="text-muted-foreground">
              Remainder:{' '}
              <span
                className={cn(
                  'font-medium',
                  remainder < 0 ? 'text-red-600' : 'text-foreground',
                )}
              >
                {fmtSizingChf(remainder)} CHF
              </span>
            </span>
          </div>
        )}

        {activateMutation.isError && (
          <div className="flex items-start gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              {activateMutation.error instanceof Error
                ? activateMutation.error.message
                : 'Activation failed. Please try again.'}
            </span>
          </div>
        )}
      </CardContent>

      <CardFooter className="gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={onBack}
          disabled={activateMutation.isPending}
        >
          <ArrowLeft className="size-3.5" />
          Back
        </Button>
        <Button
          size="sm"
          onClick={handleConfirm}
          disabled={!canConfirm}
        >
          {activateMutation.isPending ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <Check className="size-3.5" />
          )}
          {activateMutation.isPending ? 'Activating…' : 'Confirm Purchases'}
        </Button>
      </CardFooter>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Skeleton & Error
// ---------------------------------------------------------------------------

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card size="sm" key={i}>
            <CardHeader>
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-6 w-16" />
            </CardHeader>
          </Card>
        ))}
      </div>
      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-32" />
          </CardHeader>
          <CardContent className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton className="h-8 w-full" key={i} />
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function DashboardError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="size-6 text-destructive" />
        </div>
        <div>
          <p className="font-medium">Failed to load dashboard</p>
          <p className="text-sm text-muted-foreground">
            The backend may be offline. Check that the API is running on port 8000.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="size-3.5" />
          Retry
        </Button>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export function DashboardPage() {
  const queryClient = useQueryClient()

  // -- Position management state --
  const [closingId, setClosingId] = useState<number | null>(null)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [exitPrice, setExitPrice] = useState('')
  const [exitDate, setExitDate] = useState('')
  const [closeError, setCloseError] = useState<string | null>(null)

  // -- Signal generation state --
  const [signalPhase, setSignalPhase] = useState<SignalPhase>('idle')
  const [cutoffDate, setCutoffDate] = useState('')
  const [signalForSizing, setSignalForSizing] = useState<SignalOut | null>(null)
  const [activationResult, setActivationResult] = useState<ActivatePortfolioResponse | null>(null)

  // -- Investment amount (persisted in localStorage) --
  const [investmentRaw, setInvestmentRaw] = useState(() => {
    try {
      return localStorage.getItem('dashboard_investment_amount') ?? ''
    } catch {
      return ''
    }
  })

  // -- Queries --
  const dashboardQuery = useQuery({
    queryKey: ['dashboard'],
    queryFn: api.getDashboard,
    refetchInterval: 60_000,
  })

  const pnlQuery = useQuery({
    queryKey: ['portfolio-pnl'],
    queryFn: api.getPortfolioPnl,
    refetchInterval: 60_000,
  })

  const proposalQuery = useQuery({
    queryKey: ['rebalance-proposal'],
    queryFn: api.getRebalanceProposal,
    refetchInterval: 120_000,
  })

  const latestSignalQuery = useQuery({
    queryKey: ['signals', 'latest'],
    queryFn: api.getLatestSignal,
  })

  // -- Mutations --
  const invalidateAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    void queryClient.invalidateQueries({ queryKey: ['portfolio-pnl'] })
    void queryClient.invalidateQueries({ queryKey: ['portfolio-history'] })
    void queryClient.invalidateQueries({ queryKey: ['rebalance-proposal'] })
    void queryClient.invalidateQueries({ queryKey: ['signals'] })
  }, [queryClient])

  const closeMutation = useMutation({
    mutationFn: ({ id, body }: { id: number; body: { exit_price: number; exit_date?: string | null } }) =>
      api.closePosition(id, body),
    onSuccess: () => {
      invalidateAll()
      setClosingId(null)
      setExitPrice('')
      setExitDate('')
      setCloseError(null)
    },
    onError: (err: Error) => setCloseError(err.message || 'Close failed'),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, body }: { id: number; body: Record<string, unknown> }) =>
      api.updatePosition(id, body),
    onSuccess: () => {
      invalidateAll()
      setEditingId(null)
    },
  })

  const executeRebalanceMutation = useMutation({
    mutationFn: ({ rebalanceId, trades }: { rebalanceId: number; trades: ExecutedTrade[] }) =>
      api.executeRebalance({ rebalance_id: rebalanceId, executed_trades: trades }),
    onSuccess: () => invalidateAll(),
  })

  const dismissRebalanceMutation = useMutation({
    mutationFn: (rebalanceId: number) => api.dismissRebalance(rebalanceId),
    onSuccess: () => invalidateAll(),
  })

  const generateMutation = useMutation({
    mutationFn: api.generateSignal,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['signals'] })
      setSignalForSizing(data)
      setSignalPhase('sizing')
    },
  })

  // -- Derived state: prefer dashboard `positions` (full `compute_live_pnl` rows); avoid stripping fields --
  const pnl = useMemo((): PnlEntry[] => {
    const fromApi = pnlQuery.data ?? []
    const fromDash = dashboardQuery.data?.positions
    if (fromDash && fromDash.length > 0) {
      return fromDash as PnlEntry[]
    }
    return fromApi
  }, [pnlQuery.data, dashboardQuery.data?.positions])
  const investmentAmount = parseFloat(investmentRaw) || 0

  const augmentedPnl = useMemo((): PnlEntry[] => {
    if (investmentAmount <= 0 || pnl.length === 0) return pnl
    return pnl.map((p) => {
      const entryPrice = p.entry_price ?? p.current_price ?? 0
      const shares = entryPrice > 0 ? Math.floor((investmentAmount * p.weight) / entryPrice) : 0
      const entryTotal = shares * entryPrice
      const currentValue = shares * (p.current_price ?? 0)
      const pnlAbs = currentValue - entryTotal
      const pnlPct = entryTotal > 0 ? pnlAbs / entryTotal : null
      return { ...p, shares, entry_total: entryTotal, current_value: currentValue, pnl_abs: pnlAbs, pnl_pct: pnlPct }
    })
  }, [pnl, investmentAmount])

  const kpis = useMemo(() => computePortfolioKpis(augmentedPnl), [augmentedPnl])
  const returnTrend: 'up' | 'down' | 'neutral' =
    kpis.totalPnlPct == null ? 'neutral' : kpis.totalPnlPct >= 0 ? 'up' : 'down'

  const dashboardData = dashboardQuery.data
  const closingRow = closingId != null ? pnl.find((p) => p.id === closingId) : undefined

  const hasPendingProposal =
    proposalQuery.data != null && proposalQuery.data.status === 'pending'

  const hasPendingSignal =
    latestSignalQuery.data != null && latestSignalQuery.data.status === 'pending'

  const isLoading = dashboardQuery.isLoading || pnlQuery.isLoading
  const isError = !isLoading && (dashboardQuery.isError || pnlQuery.isError)

  // -- Position handlers --
  const openClose = (id: number) => {
    setCloseError(null)
    setEditingId(null)
    setClosingId(id)
    const row = pnl.find((p) => p.id === id)
    setExitPrice(row?.current_price != null ? String(row.current_price) : '')
    setExitDate('')
  }

  const submitClose = () => {
    if (closingId == null) return
    const price = Number.parseFloat(exitPrice.replace(',', '.'))
    if (!Number.isFinite(price) || price <= 0) {
      setCloseError('Enter a valid exit price.')
      return
    }
    setCloseError(null)
    closeMutation.mutate({
      id: closingId,
      body: { exit_price: price, exit_date: exitDate.trim() || null },
    })
  }

  const handleEditPosition = (positionId: number, fields: Record<string, unknown>) => {
    if (Object.keys(fields).length === 0) {
      setClosingId(null)
      setEditingId(positionId)
      return
    }
    updateMutation.mutate({ id: positionId, body: fields })
  }

  // -- Signal generation handlers --
  const handleStartGenerate = () => {
    setCutoffDate('')
    generateMutation.reset()
    setSignalPhase('confirming')
  }

  const handleConfirmGenerate = () => {
    generateMutation.mutate({ cutoff_date: cutoffDate || undefined })
  }

  const handleActivatePending = () => {
    if (latestSignalQuery.data) {
      setSignalForSizing(latestSignalQuery.data)
      setSignalPhase('sizing')
    }
  }

  const handleActivationSuccess = (result: ActivatePortfolioResponse) => {
    setActivationResult(result)
    invalidateAll()
    setSignalPhase('result')
  }

  const handleDismissSignal = () => {
    setSignalPhase('idle')
    setSignalForSizing(null)
    setActivationResult(null)
    generateMutation.reset()
  }

  // -- Investment input handler --
  const handleInvestmentChange = (raw: string) => {
    setInvestmentRaw(raw)
    try {
      localStorage.setItem('dashboard_investment_amount', raw)
    } catch { /* quota exceeded */ }
  }

  // -- Refresh handler --
  const handleRefresh = () => {
    void dashboardQuery.refetch()
    void pnlQuery.refetch()
    void proposalQuery.refetch()
    void latestSignalQuery.refetch()
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Lag60-SA model · Semi-annual rebalancing · Portfolio, signals and alerts at a glance
          </p>
        </div>
        <div className="flex items-center gap-2">
          {signalPhase === 'idle' && (
            <Button onClick={handleStartGenerate} disabled={isLoading} size="sm">
              <Sparkles className="size-3.5" />
              Generate New Signal
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isLoading}
          >
            <RefreshCw className={cn('size-3.5', isLoading && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {isLoading && <DashboardSkeleton />}
      {isError && <DashboardError onRetry={handleRefresh} />}

      {!isLoading && !isError && (
        <>
          {/* Investment amount input */}
          <div className="flex items-center gap-3">
            <label
              className="text-sm font-medium text-muted-foreground whitespace-nowrap"
              htmlFor="dashboard-investment"
            >
              Investment
            </label>
            <div className="relative">
              <input
                id="dashboard-investment"
                type="number"
                min={0}
                step={1000}
                placeholder="e.g. 100 000"
                value={investmentRaw}
                onChange={(e) => handleInvestmentChange(e.target.value)}
                className="flex h-8 w-48 rounded-md border border-border bg-background px-3 pr-10 text-sm tabular-nums outline-none focus:ring-2 focus:ring-ring/50"
              />
              <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
                CHF
              </span>
            </div>
            {investmentAmount > 0 && kpis.count > 0 && (
              <span className="text-xs text-muted-foreground tabular-nums">
                Allocated: {fmtChf(kpis.totalInvested)} CHF
                {' · '}
                Remainder: {fmtChf(investmentAmount - kpis.totalInvested)} CHF
              </span>
            )}
          </div>

          {/* KPI row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard
              title="Portfolio Value"
              value={
                kpis.totalCurrentValue > 0
                  ? `${fmtChf(kpis.totalCurrentValue)} CHF`
                  : '—'
              }
              subtitle={
                investmentAmount > 0
                  ? `Investment: ${fmtChf(investmentAmount)} CHF`
                  : kpis.totalInvested > 0
                    ? `Invested: ${fmtChf(kpis.totalInvested)} CHF`
                    : `${kpis.count} active positions`
              }
              icon={Wallet}
            />

            <KpiCard
              title="Total P&L"
              value={kpis.totalPnlPct != null ? fmtPct(kpis.totalPnlPct) : '—'}
              subtitle={
                kpis.totalInvested > 0
                  ? `${kpis.totalPnlAbs >= 0 ? '+' : ''}${fmtChf(kpis.totalPnlAbs)} CHF`
                  : `${kpis.count} active positions`
              }
              icon={TrendingUp}
              trend={returnTrend}
            />

            <KpiCard
              title="Best / Worst"
              value={kpis.best ? kpis.best.ticker : '—'}
              subtitle={
                kpis.best && kpis.worst
                  ? `${fmtPct(kpis.best.pnl_pct)} · Worst: ${kpis.worst.ticker} ${fmtPct(kpis.worst.pnl_pct)}`
                  : 'No positions with P&L'
              }
              icon={ArrowUp}
              trend={
                kpis.best?.pnl_pct != null
                  ? kpis.best.pnl_pct >= 0
                    ? 'up'
                    : 'down'
                  : 'neutral'
              }
            />

            <KpiCard
              title="Next Rebalancing"
              value={dashboardData ? `${dashboardData.next_rebalancing.days_until}d` : '—'}
              subtitle={
                dashboardData
                  ? new Date(dashboardData.next_rebalancing.date).toLocaleDateString('de-CH', {
                      day: 'numeric',
                      month: 'short',
                      year: 'numeric',
                    })
                  : undefined
              }
              icon={CalendarClock}
            />
          </div>

          {/* Signal generation wizard (shown when active) */}
          {signalPhase === 'confirming' && (
            <div className="space-y-3">
              <StepIndicator step={1} />
              <Card className="border-primary/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="size-4 text-primary" />
                    Generate New Signal
                  </CardTitle>
                  <CardDescription>
                    A new signal will be generated from the latest market data. You
                    can configure your investment before activating.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-1.5">
                    <label
                      className="text-xs font-medium text-muted-foreground"
                      htmlFor="cutoff"
                    >
                      Cutoff date (optional — defaults to today)
                    </label>
                    <input
                      id="cutoff"
                      type="date"
                      value={cutoffDate}
                      onChange={(e) => setCutoffDate(e.target.value)}
                      className="flex h-8 w-full max-w-xs rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
                    />
                  </div>
                  {generateMutation.isError && (
                    <div className="flex items-start gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                      <span>
                        {generateMutation.error instanceof Error
                          ? generateMutation.error.message
                          : 'Signal generation failed. Please try again.'}
                      </span>
                    </div>
                  )}
                </CardContent>
                <CardFooter className="gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDismissSignal}
                    disabled={generateMutation.isPending}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleConfirmGenerate}
                    disabled={generateMutation.isPending}
                  >
                    {generateMutation.isPending ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Zap className="size-3.5" />
                    )}
                    {generateMutation.isPending ? 'Generating…' : 'Generate Signal'}
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}

          {signalPhase === 'sizing' && signalForSizing && (
            <div className="space-y-3">
              <StepIndicator step={2} />
              <SizingCard
                signal={signalForSizing}
                onBack={() => setSignalPhase('confirming')}
                onSuccess={handleActivationSuccess}
              />
            </div>
          )}

          {signalPhase === 'result' && activationResult && (
            <Card className="border-emerald-500/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Check className="size-4 text-emerald-600" />
                  Portfolio Activated
                </CardTitle>
                <CardDescription>
                  Signal #{activationResult.signal_id} is now your active portfolio
                  with {activationResult.positions_activated} positions (
                  {fmtSizingChf(activationResult.investment_amount)} CHF invested).
                </CardDescription>
              </CardHeader>
              <CardFooter>
                <Button variant="outline" size="sm" onClick={handleDismissSignal}>
                  <Check className="size-3.5" />
                  Done
                </Button>
              </CardFooter>
            </Card>
          )}

          {/* Pending signal notice */}
          {signalPhase === 'idle' && hasPendingSignal && pnl.length === 0 && latestSignalQuery.data && (
            <Card className="border-blue-300 bg-blue-50/50 dark:border-blue-700 dark:bg-blue-950/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Sparkles className="size-4 text-blue-600" />
                  Pending Signal
                </CardTitle>
                <CardDescription className="flex flex-wrap items-center gap-2">
                  <span>
                    Signal #{latestSignalQuery.data.id} from {latestSignalQuery.data.cutoff_date} has not been activated yet.
                    Configure your trades to start tracking your portfolio.
                  </span>
                  {latestSignalQuery.data.requested_top_n != null &&
                    latestSignalQuery.data.portfolio.length <
                      latestSignalQuery.data.requested_top_n && (
                      <Badge
                        variant="outline"
                        className="border-amber-500/55 text-[0.65rem] text-amber-900 dark:text-amber-100"
                        title="Fewer names than requested: some tickers lack sufficient OHLCV or model features at this cutoff."
                      >
                        <Info className="mr-0.5 inline size-3 opacity-80" aria-hidden />
                        {latestSignalQuery.data.portfolio.length} of{' '}
                        {latestSignalQuery.data.requested_top_n} positions
                      </Badge>
                    )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button size="sm" onClick={handleActivatePending}>
                  <Zap className="size-3.5" />
                  Configure Trades
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Rebalancing Banner */}
          {hasPendingProposal && proposalQuery.data && (
            <RebalancingBanner
              proposal={proposalQuery.data}
              onExecute={(trades) =>
                executeRebalanceMutation.mutate({
                  rebalanceId: proposalQuery.data!.id,
                  trades,
                })
              }
              onDismiss={() => dismissRebalanceMutation.mutate(proposalQuery.data!.id)}
              isExecuting={executeRebalanceMutation.isPending}
              isDismissing={dismissRebalanceMutation.isPending}
            />
          )}

          {/* Main content: Active Portfolio */}
          <div className="grid gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wallet className="size-4 opacity-70" />
                  Active Portfolio
                </CardTitle>
                {dashboardData?.signal && (
                  <CardDescription className="flex flex-wrap items-center gap-x-1 gap-y-1">
                    Signal from {dashboardData.signal.cutoff_date}
                    {' · '}
                    <Badge variant="secondary" className="ml-1 text-[0.65rem]">
                      {dashboardData.signal.status}
                    </Badge>
                    {dashboardData.signal.n_positions != null &&
                      dashboardData.signal.requested_top_n != null &&
                      dashboardData.signal.n_positions <
                        dashboardData.signal.requested_top_n && (
                        <Badge
                          variant="outline"
                          className="border-amber-500/55 text-[0.65rem] text-amber-900 dark:text-amber-100"
                          title="Fewer names than requested: some tickers lack sufficient OHLCV or model features at this cutoff."
                        >
                          <Info className="mr-0.5 inline size-3 opacity-80" aria-hidden />
                          {dashboardData.signal.n_positions} of{' '}
                          {dashboardData.signal.requested_top_n} positions
                        </Badge>
                      )}
                    {kpis.count > 0 && dashboardData.signal.status === 'active' && (
                      <>
                        {' · '}
                        <span className="tabular-nums">{kpis.count} positions</span>
                        {kpis.positionsWithValue > 0 && (
                          <span className="text-muted-foreground">
                            {' '}({kpis.positionsWithValue} with live value)
                          </span>
                        )}
                      </>
                    )}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-4">
                <PortfolioTable
                  positions={augmentedPnl}
                  onClosePosition={openClose}
                  onEditPosition={handleEditPosition}
                  editingId={editingId}
                  onCancelEdit={() => setEditingId(null)}
                  isSaving={updateMutation.isPending}
                />

                {/* Close position dialog */}
                {closingId != null && closingRow && (
                  <div className="rounded-lg border border-border bg-muted/30 p-4">
                    <p className="mb-3 text-sm font-medium">
                      Close <span className="font-mono">{closingRow.ticker}</span>
                      {closingRow.shares != null && (
                        <span className="text-muted-foreground"> ({closingRow.shares} shares)</span>
                      )}
                    </p>
                    <div className="grid gap-3 sm:grid-cols-2 sm:items-end">
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground" htmlFor="exit-price">
                          Exit price
                        </label>
                        <input
                          id="exit-price"
                          type="text"
                          inputMode="decimal"
                          className={inputClass}
                          value={exitPrice}
                          onChange={(e) => setExitPrice(e.target.value)}
                          placeholder="e.g. 120.50"
                        />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground" htmlFor="exit-date">
                          Exit date (optional)
                        </label>
                        <input
                          id="exit-date"
                          type="date"
                          className={inputClass}
                          value={exitDate}
                          onChange={(e) => setExitDate(e.target.value)}
                        />
                      </div>
                    </div>
                    {closeError && <p className="mt-2 text-sm text-destructive">{closeError}</p>}
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button size="sm" onClick={submitClose} disabled={closeMutation.isPending}>
                        {closeMutation.isPending ? 'Closing…' : 'Confirm close'}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setClosingId(null)
                          setCloseError(null)
                        }}
                        disabled={closeMutation.isPending}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
