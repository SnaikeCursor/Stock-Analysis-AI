import { Fragment, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  ChevronDown,
  History,
  Loader2,
  Radio,
  RefreshCw,
  Sparkles,
  Zap,
} from 'lucide-react'

import { api } from '@/lib/api'
import type {
  SignalOut,
  ActivatePortfolioBody,
  ActivatePortfolioResponse,
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
import { SignalCard } from '@/components/SignalCard'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Phase = 'idle' | 'confirming' | 'sizing' | 'result'

type EditableRow = {
  ticker: string
  weight: number
  predicted_return: number
  currentPrice: number
  shares: number
  entryPrice: number
  entryDate: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const INPUT_CLS =
  'flex h-7 w-full rounded border border-border bg-background px-2 text-sm tabular-nums outline-none focus:ring-2 focus:ring-ring/50'

function fmtPct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function fmtChf(v: number) {
  return v.toLocaleString('de-CH', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function todayIso() {
  return new Date().toISOString().slice(0, 10)
}

// ---------------------------------------------------------------------------
// Step Indicator
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Sizing Card (Step 2 — investment amount + editable positions)
// ---------------------------------------------------------------------------

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
        {/* Investment amount */}
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

        {/* Positions table */}
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
                      {fmtPct(r.weight)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {r.currentPrice > 0 ? fmtChf(r.currentPrice) : '—'}
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
                            shares: Math.max(
                              0,
                              parseInt(e.target.value) || 0,
                            ),
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
                            entryPrice: Math.max(
                              0,
                              parseFloat(e.target.value) || 0,
                            ),
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
                      {amount > 0 ? fmtChf(amount) : '—'}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>

        {/* Totals */}
        {investment > 0 && (
          <div className="flex items-center gap-4 text-sm">
            <span className="text-muted-foreground">
              Total:{' '}
              <span className="font-medium text-foreground">
                {fmtChf(totalInvested)} CHF
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
                {fmtChf(remainder)} CHF
              </span>
            </span>
          </div>
        )}

        {/* Activation error */}
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
// Skeleton / Error / Empty
// ---------------------------------------------------------------------------

function SignalsSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-3 w-48" />
      </CardHeader>
      <CardContent className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton className="h-8 w-full" key={i} />
        ))}
      </CardContent>
    </Card>
  )
}

function SignalsError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="size-6 text-destructive" />
        </div>
        <div>
          <p className="font-medium">Failed to load signals</p>
          <p className="text-sm text-muted-foreground">
            The backend may be offline. Check that the API is running.
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

function EmptyState({ onGenerate }: { onGenerate: () => void }) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-4 py-14 text-center">
        <div className="flex size-14 items-center justify-center rounded-full bg-primary/10">
          <Radio className="size-7 text-primary" />
        </div>
        <div>
          <p className="text-lg font-medium">No signals yet</p>
          <p className="mt-1 max-w-sm text-sm text-muted-foreground">
            Generate your first trading signal using the Lag60-SA regression model
            with semi-annual rebalancing from the SPI universe.
          </p>
        </div>
        <Button onClick={onGenerate}>
          <Sparkles className="size-3.5" />
          Generate First Signal
        </Button>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// History Section
// ---------------------------------------------------------------------------

function HistorySection({ signals }: { signals: SignalOut[] }) {
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set())

  const toggle = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  if (signals.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="size-4 opacity-70" />
          Signal History
        </CardTitle>
        <CardDescription>
          {signals.length} signal{signals.length !== 1 ? 's' : ''} generated
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8" />
              <TableHead>Date</TableHead>
              <TableHead className="text-right">Positions</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {signals.map((sig) => {
              const isOpen = expandedIds.has(sig.id)
              return (
                <Fragment key={sig.id}>
                  <TableRow
                    className="cursor-pointer"
                    onClick={() => toggle(sig.id)}
                  >
                    <TableCell>
                      <ChevronDown
                        className={cn(
                          'size-3.5 text-muted-foreground transition-transform',
                          isOpen && 'rotate-180',
                        )}
                      />
                    </TableCell>
                    <TableCell className="font-medium tabular-nums">
                      {sig.cutoff_date}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {sig.portfolio.length}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-[0.65rem]">
                        {sig.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">
                      {sig.created_at
                        ? formatDistanceToNow(new Date(sig.created_at), {
                            addSuffix: true,
                          })
                        : '—'}
                    </TableCell>
                  </TableRow>
                  {isOpen && (
                    <TableRow className="hover:bg-transparent">
                      <TableCell colSpan={5} className="bg-muted/30 py-3">
                        <div className="flex flex-wrap gap-2 px-2">
                          {sig.portfolio.map((p) => (
                            <div
                              key={p.ticker}
                              className="flex items-center gap-1.5 rounded-md border bg-background px-2 py-1 text-xs"
                            >
                              <span className="font-medium">{p.ticker}</span>
                              <span className="tabular-nums text-muted-foreground">
                                {fmtPct(p.weight)}
                              </span>
                              <Separator
                                orientation="vertical"
                                className="h-3"
                              />
                              <span className="tabular-nums text-muted-foreground">
                                {p.predicted_return >= 0 ? '+' : ''}{p.predicted_return.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </Fragment>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export function SignalsPage() {
  const queryClient = useQueryClient()
  const [phase, setPhase] = useState<Phase>('idle')
  const [cutoffDate, setCutoffDate] = useState('')
  const [activationResult, setActivationResult] =
    useState<ActivatePortfolioResponse | null>(null)

  const latestQuery = useQuery({
    queryKey: ['signals', 'latest'],
    queryFn: api.getLatestSignal,
  })

  const historyQuery = useQuery({
    queryKey: ['signals', 'history'],
    queryFn: api.getSignalHistory,
  })

  const generateMutation = useMutation({
    mutationFn: api.generateSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] })
      setPhase('sizing')
    },
  })

  const isLoading = latestQuery.isLoading || historyQuery.isLoading
  const isError = !isLoading && latestQuery.isError && historyQuery.isError
  const hasSignals =
    latestQuery.data != null || (historyQuery.data?.length ?? 0) > 0

  const handleStartGenerate = () => {
    setCutoffDate('')
    generateMutation.reset()
    setPhase('confirming')
  }

  const handleConfirmGenerate = () => {
    generateMutation.mutate({
      cutoff_date: cutoffDate || undefined,
    })
  }

  const handleActivationSuccess = (result: ActivatePortfolioResponse) => {
    setActivationResult(result)
    queryClient.invalidateQueries({ queryKey: ['signals'] })
    queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    queryClient.invalidateQueries({ queryKey: ['portfolio'] })
    setPhase('result')
  }

  const handleDismiss = () => {
    setPhase('idle')
    setActivationResult(null)
    generateMutation.reset()
  }

  const activeSignalId = latestQuery.data?.id
  const pastSignals = (historyQuery.data ?? []).filter(
    (s) => s.id !== activeSignalId,
  )

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Signals</h1>
          <p className="text-sm text-muted-foreground">
            Generate and review trading signals.
          </p>
        </div>
        {phase === 'idle' && hasSignals && (
          <Button onClick={handleStartGenerate} disabled={isLoading}>
            <Sparkles className="size-3.5" />
            Generate New Signal
          </Button>
        )}
      </div>

      {/* Step 1: Generate signal */}
      {phase === 'confirming' && (
        <>
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
                onClick={() => setPhase('idle')}
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
                {generateMutation.isPending
                  ? 'Generating…'
                  : 'Generate Signal'}
              </Button>
            </CardFooter>
          </Card>
        </>
      )}

      {/* Step 2: Configure trades */}
      {phase === 'sizing' && generateMutation.data && (
        <>
          <StepIndicator step={2} />
          <SizingCard
            signal={generateMutation.data}
            onBack={() => setPhase('confirming')}
            onSuccess={handleActivationSuccess}
          />
        </>
      )}

      {/* Result */}
      {phase === 'result' && activationResult && (
        <Card className="border-emerald-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Check className="size-4 text-emerald-600" />
              Portfolio Activated
            </CardTitle>
            <CardDescription>
              Signal #{activationResult.signal_id} is now your active portfolio
              with {activationResult.positions_activated} positions (
              {fmtChf(activationResult.investment_amount)} CHF invested).
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button variant="outline" size="sm" onClick={handleDismiss}>
              <Check className="size-3.5" />
              Done
            </Button>
          </CardFooter>
        </Card>
      )}

      {/* Main content — idle phase */}
      {phase === 'idle' && (
        <>
          {isLoading && <SignalsSkeleton />}
          {isError && (
            <SignalsError
              onRetry={() => {
                latestQuery.refetch()
                historyQuery.refetch()
              }}
            />
          )}
          {!isLoading && !isError && !hasSignals && (
            <EmptyState onGenerate={handleStartGenerate} />
          )}
          {!isLoading && latestQuery.data && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Radio className="size-4 opacity-70" />
                  {latestQuery.data.status === 'pending'
                    ? 'Pending Signal'
                    : 'Active Signal'}
                </CardTitle>
                <CardDescription>
                  Signal #{latestQuery.data.id}
                  {latestQuery.data.created_at && (
                    <>
                      {' · '}
                      {formatDistanceToNow(
                        new Date(latestQuery.data.created_at),
                        { addSuffix: true },
                      )}
                    </>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SignalCard signal={latestQuery.data} />
              </CardContent>
            </Card>
          )}
          {!isLoading && pastSignals.length > 0 && (
            <HistorySection signals={pastSignals} />
          )}
        </>
      )}
    </div>
  )
}
