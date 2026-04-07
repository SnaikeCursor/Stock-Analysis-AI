import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { format } from 'date-fns'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ArrowRightLeft,
  Check,
  History,
  RefreshCw,
  TrendingUp,
  Wallet,
  X,
} from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'

import { PortfolioTable } from '@/components/PortfolioTable'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  api,
  type ExecutedTrade,
  type PnlEntry,
  type PositionDetail,
  type RebalanceProposal,
  type RebalanceInstructionRow,
} from '@/lib/api'
import { cn } from '@/lib/utils'

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

function fmtWeight(v: number): string {
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

  return {
    totalInvested,
    totalCurrentValue,
    totalPnlAbs,
    totalPnlPct,
    count: positions.length,
    positionsWithValue,
  }
}

// ---------------------------------------------------------------------------
// Skeleton & Error
// ---------------------------------------------------------------------------

function PortfolioSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card size="sm" key={i}>
            <CardHeader>
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-7 w-20" />
            </CardHeader>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-40" />
        </CardHeader>
        <CardContent>
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton className="mb-2 h-8 w-full" key={i} />
          ))}
        </CardContent>
      </Card>
    </div>
  )
}

function PortfolioError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="size-6 text-destructive" />
        </div>
        <div>
          <p className="font-medium">Failed to load portfolio</p>
          <p className="text-sm text-muted-foreground">
            Check that the API is running (e.g. port 8000).
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
// Rebalancing Banner
// ---------------------------------------------------------------------------

const inputClass =
  'flex h-8 w-full min-w-0 rounded-md border border-input bg-transparent px-2 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'

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
// Historical Block
// ---------------------------------------------------------------------------

function PositionStatus({ p }: { p: PositionDetail }) {
  const closed = p.exit_date != null && p.exit_date !== ''
  return (
    <Badge variant={closed ? 'secondary' : 'outline'} className="text-[0.65rem]">
      {closed ? 'Closed' : 'Open'}
    </Badge>
  )
}

function HistoricalBlock({
  signalId,
  cutoffDate,
  status,
  positions,
}: {
  signalId: number
  cutoffDate: string
  status: string
  positions: PositionDetail[]
}) {
  const closedCount = positions.filter((p) => p.exit_date).length

  return (
    <Card size="sm">
      <CardHeader className="border-b border-border/60 pb-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle className="text-sm">Signal #{signalId}</CardTitle>
            <CardDescription className="mt-1 flex flex-wrap items-center gap-2">
              <span>Cutoff {cutoffDate}</span>
              <Badge variant="secondary" className="text-[0.65rem]">
                {status}
              </Badge>
            </CardDescription>
          </div>
          <span className="text-[0.7rem] text-muted-foreground tabular-nums">
            {closedCount}/{positions.length} closed
          </span>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        {positions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No positions recorded.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ticker</TableHead>
                <TableHead className="text-right">Shares</TableHead>
                <TableHead className="text-right">Weight</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Exit</TableHead>
                <TableHead className="text-right">Exit date</TableHead>
                <TableHead className="text-right">P&amp;L</TableHead>
                <TableHead className="w-[1%]"> </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((p) => {
                const pnl = p.pnl_pct
                const isPositive = pnl != null && pnl > 0
                const isNegative = pnl != null && pnl < 0
                return (
                  <TableRow key={p.id}>
                    <TableCell className="font-medium">{p.ticker}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {p.shares != null ? p.shares : '—'}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{fmtWeight(p.weight)}</TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {fmtPrice(p.entry_price)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {p.exit_price != null ? fmtPrice(p.exit_price) : '—'}
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">{fmtDate(p.exit_date)}</TableCell>
                    <TableCell className="text-right">
                      <span
                        className={cn(
                          'inline-flex items-center gap-0.5 tabular-nums font-medium',
                          isPositive && 'text-emerald-600',
                          isNegative && 'text-red-600',
                        )}
                      >
                        {isPositive && <ArrowUp className="size-3" />}
                        {isNegative && <ArrowDown className="size-3" />}
                        {fmtPct(pnl)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <PositionStatus p={p} />
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export function PortfolioPage() {
  const queryClient = useQueryClient()
  const [closingId, setClosingId] = useState<number | null>(null)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [exitPrice, setExitPrice] = useState('')
  const [exitDate, setExitDate] = useState('')
  const [closeError, setCloseError] = useState<string | null>(null)

  // Queries
  const {
    data: pnl,
    isLoading: pnlLoading,
    isError: pnlError,
    refetch: refetchPnl,
  } = useQuery({
    queryKey: ['portfolio-pnl'],
    queryFn: api.getPortfolioPnl,
    refetchInterval: 60_000,
  })

  const {
    data: history,
    isLoading: historyLoading,
    isError: historyError,
    refetch: refetchHistory,
  } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: api.getPortfolioHistory,
  })

  const {
    data: rebalanceProposal,
    isLoading: proposalLoading,
    refetch: refetchProposal,
  } = useQuery({
    queryKey: ['rebalance-proposal'],
    queryFn: api.getRebalanceProposal,
    refetchInterval: 120_000,
  })

  // Mutations
  const invalidateAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['portfolio-pnl'] })
    void queryClient.invalidateQueries({ queryKey: ['portfolio-history'] })
    void queryClient.invalidateQueries({ queryKey: ['rebalance-proposal'] })
    void queryClient.invalidateQueries({ queryKey: ['dashboard'] })
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

  // Derived state
  const kpis = useMemo(() => computePortfolioKpis(pnl ?? []), [pnl])
  const returnTrend: 'up' | 'down' | 'neutral' =
    kpis.totalPnlPct == null ? 'neutral' : kpis.totalPnlPct >= 0 ? 'up' : 'down'

  const closingRow = closingId != null ? pnl?.find((p) => p.id === closingId) : undefined

  const hasPendingProposal =
    rebalanceProposal != null &&
    rebalanceProposal.status === 'pending'

  // Handlers
  const openClose = (id: number) => {
    setCloseError(null)
    setEditingId(null)
    setClosingId(id)
    const row = pnl?.find((p) => p.id === id)
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

  const loading = pnlLoading || historyLoading || proposalLoading
  const error = pnlError || historyError

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Portfolio</h1>
          <p className="text-sm text-muted-foreground">
            Active positions with live P&amp;L and historical trades by signal
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void refetchPnl()
            void refetchHistory()
            void refetchProposal()
          }}
          disabled={loading}
        >
          <RefreshCw className={cn('size-3.5', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {loading && <PortfolioSkeleton />}
      {error && (
        <PortfolioError
          onRetry={() => {
            void refetchPnl()
            void refetchHistory()
          }}
        />
      )}

      {!loading && !error && (
        <>
          {/* KPI Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card size="sm">
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <Wallet className="size-3.5 opacity-70" />
                  Portfolio Value
                </CardDescription>
                <CardTitle className="text-xl tabular-nums">
                  {kpis.totalCurrentValue > 0
                    ? `${fmtChf(kpis.totalCurrentValue)} CHF`
                    : '—'}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  Invested: {kpis.totalInvested > 0 ? `${fmtChf(kpis.totalInvested)} CHF` : '—'}
                </p>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <TrendingUp className="size-3.5 opacity-70" />
                  Total P&amp;L
                </CardDescription>
                <CardTitle
                  className={cn(
                    'text-xl tabular-nums',
                    returnTrend === 'up' && 'text-emerald-600',
                    returnTrend === 'down' && 'text-red-600',
                  )}
                >
                  <span className="inline-flex items-center gap-1">
                    {returnTrend === 'up' && <ArrowUp className="size-4" />}
                    {returnTrend === 'down' && <ArrowDown className="size-4" />}
                    {kpis.totalPnlPct != null
                      ? fmtPct(kpis.totalPnlPct)
                      : '—'}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className={cn(
                  'text-xs font-medium tabular-nums',
                  kpis.totalPnlAbs > 0 && 'text-emerald-600',
                  kpis.totalPnlAbs < 0 && 'text-red-600',
                )}>
                  {kpis.totalInvested > 0
                    ? `${kpis.totalPnlAbs >= 0 ? '+' : ''}${fmtChf(kpis.totalPnlAbs)} CHF`
                    : 'Based on invested amounts'}
                </p>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardHeader>
                <CardDescription>Active positions</CardDescription>
                <CardTitle className="text-xl tabular-nums">{pnl?.length ?? 0}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  {kpis.positionsWithValue > 0
                    ? `${kpis.positionsWithValue} with live value`
                    : 'No live marks'}
                </p>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <History className="size-3.5 opacity-70" />
                  Signals in history
                </CardDescription>
                <CardTitle className="text-xl tabular-nums">{history?.length ?? 0}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">Saved signals with positions</p>
              </CardContent>
            </Card>
          </div>

          {/* Rebalancing Banner */}
          {hasPendingProposal && rebalanceProposal && (
            <RebalancingBanner
              proposal={rebalanceProposal}
              onExecute={(trades) =>
                executeRebalanceMutation.mutate({
                  rebalanceId: rebalanceProposal.id,
                  trades,
                })
              }
              onDismiss={() => dismissRebalanceMutation.mutate(rebalanceProposal.id)}
              isExecuting={executeRebalanceMutation.isPending}
              isDismissing={dismissRebalanceMutation.isPending}
            />
          )}

          {/* Active Positions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wallet className="size-4 opacity-70" />
                Active positions
              </CardTitle>
              <CardDescription>
                Shares, invested amounts, and live P&amp;L — click pencil to edit, refresh every 60s
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <PortfolioTable
                positions={pnl ?? []}
                onClosePosition={openClose}
                onEditPosition={handleEditPosition}
                editingId={editingId}
                onCancelEdit={() => setEditingId(null)}
                isSaving={updateMutation.isPending}
              />

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

          {/* Historical Positions */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <History className="size-4 text-muted-foreground" />
              <h2 className="text-lg font-medium tracking-tight">Historical positions</h2>
            </div>
            <p className="text-sm text-muted-foreground">
              All saved signals, newest first. Open rows are still active; closed rows show realized
              P&amp;L.
            </p>

            {!history?.length ? (
              <Card>
                <CardContent className="py-10 text-center text-sm text-muted-foreground">
                  No signal history yet. Generate and save a signal to see positions here.
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {history.map((sig) => (
                  <HistoricalBlock
                    key={sig.signal_id}
                    signalId={sig.signal_id}
                    cutoffDate={sig.cutoff_date}
                    status={sig.status}
                    positions={sig.positions}
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
