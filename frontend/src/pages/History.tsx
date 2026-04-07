import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  AlertTriangle,
  Archive,
  ArrowDownUp,
  BarChart3,
  CalendarRange,
  ChevronDown,
  ChevronRight,
  DollarSign,
  FlaskConical,
  Info,
  Play,
  RefreshCw,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'

import { api } from '@/lib/api'
import type {
  HistoryPerformanceResponse,
  PortfolioHistorySignal,
  QuarterlyDetail,
  SimulateResponse,
  SimulateSummary,
  SimulateTimelinePoint,
  SimulateTransaction,
} from '@/lib/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const STRATEGY_CLR = 'oklch(0.65 0.17 162)'
const BENCHMARK_CLR = 'oklch(0.6 0.12 250)'

function fmtPct(v: number, d = 1) {
  return `${(v * 100).toFixed(d)}%`
}

function heatCellBg(pct: number): string {
  const intensity = Math.min(Math.abs(pct) / 15, 1)
  const chroma = intensity * 0.14
  const hue = pct >= 0 ? 150 : 25
  const lightness = 0.92 - intensity * 0.1
  return `oklch(${lightness.toFixed(2)} ${chroma.toFixed(3)} ${hue})`
}

function fmtChf(v: number): string {
  return new Intl.NumberFormat('de-CH', { maximumFractionDigits: 0 }).format(v)
}

function fmtChfAxis(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 10_000) return `${(v / 1_000).toFixed(0)}K`
  return String(Math.round(v))
}

// ---------------------------------------------------------------------------
// KPI Card (mirrors Dashboard pattern)
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
        <div
          className={cn(
            'text-xl font-medium tabular-nums',
            trend === 'up' && 'text-emerald-600',
            trend === 'down' && 'text-red-600',
          )}
        >
          {value}
        </div>
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
// Yearly Comparison — grouped bar chart
// ---------------------------------------------------------------------------

function YearlyComparisonChart({ data }: { data: HistoryPerformanceResponse }) {
  const chartData = useMemo(
    () =>
      Object.entries(data.per_year)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([year, m]) => ({
          year,
          Strategy: m.long_only.cumulative_return * 100,
          Benchmark: m.benchmark.cumulative_return * 100,
        })),
    [data],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="size-4 opacity-70" />
          Yearly Returns
        </CardTitle>
        <CardDescription>Strategy vs. SPI benchmark (annual return)</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barGap={2} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis dataKey="year" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={48}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <div className="rounded-lg border bg-card px-3 py-2 text-card-foreground shadow-lg">
                      <p className="mb-1 text-sm font-medium">{label}</p>
                      {payload.map((e) => (
                        <p key={String(e.name)} className="text-xs" style={{ color: String(e.color) }}>
                          {e.name}: {Number(e.value).toFixed(2)}%
                        </p>
                      ))}
                    </div>
                  )
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <ReferenceLine y={0} stroke="var(--foreground)" strokeOpacity={0.2} />
              <Bar name="Strategy" dataKey="Strategy" fill={STRATEGY_CLR} radius={[3, 3, 0, 0]} />
              <Bar name="Benchmark" dataKey="Benchmark" fill={BENCHMARK_CLR} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Cumulative Growth — area chart (CHF 100 invested)
// ---------------------------------------------------------------------------

function CumulativeReturnChart({ data }: { data: HistoryPerformanceResponse }) {
  const chartData = useMemo(() => {
    const years = Object.entries(data.per_year).sort(([a], [b]) => Number(a) - Number(b))
    if (years.length === 0) return []

    let sCum = 100
    let bCum = 100
    const pts = [{ year: `${Number(years[0][0]) - 1}`, Strategy: 100, Benchmark: 100 }]

    for (const [year, m] of years) {
      sCum *= 1 + m.long_only.cumulative_return
      bCum *= 1 + m.benchmark.cumulative_return
      pts.push({
        year,
        Strategy: Math.round(sCum * 100) / 100,
        Benchmark: Math.round(bCum * 100) / 100,
      })
    }
    return pts
  }, [data])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="size-4 opacity-70" />
          Cumulative Growth
        </CardTitle>
        <CardDescription>Growth of CHF 100 invested at the start</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="grad-strat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={STRATEGY_CLR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={STRATEGY_CLR} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="grad-bench" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={BENCHMARK_CLR} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={BENCHMARK_CLR} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis dataKey="year" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={50}
                tickFormatter={(v: number) => `${v.toFixed(0)}`}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <div className="rounded-lg border bg-card px-3 py-2 text-card-foreground shadow-lg">
                      <p className="mb-1 text-sm font-medium">{label}</p>
                      {payload.map((e) => (
                        <p key={String(e.name)} className="text-xs" style={{ color: String(e.color) }}>
                          {e.name}: CHF {Number(e.value).toFixed(2)}
                        </p>
                      ))}
                    </div>
                  )
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <ReferenceLine y={100} stroke="var(--foreground)" strokeOpacity={0.15} strokeDasharray="4 4" />
              <Area
                type="monotone"
                name="Strategy"
                dataKey="Strategy"
                stroke={STRATEGY_CLR}
                strokeWidth={2}
                fill="url(#grad-strat)"
              />
              <Area
                type="monotone"
                name="Benchmark"
                dataKey="Benchmark"
                stroke={BENCHMARK_CLR}
                strokeWidth={2}
                fill="url(#grad-bench)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Quarterly Heatmap — color-coded return grid
// ---------------------------------------------------------------------------

function QuarterlyHeatmap({ data }: { data: QuarterlyDetail[] }) {
  const { years, quarters, grid } = useMemo(() => {
    const ySet = new Set<number>()
    const qSet = new Set<string>()
    const g = new Map<string, QuarterlyDetail>()

    for (const d of data) {
      ySet.add(d.year)
      qSet.add(d.quarter)
      g.set(`${d.year}-${d.quarter}`, d)
    }

    return {
      years: [...ySet].sort((a, b) => a - b),
      quarters: [...qSet].sort(),
      grid: g,
    }
  }, [data])

  if (years.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CalendarRange className="size-4 opacity-70" />
          Quarterly Detail
        </CardTitle>
        <CardDescription>
          Per-quarter returns (hover for turnover, costs &amp; Sharpe)
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-3 font-medium">Year</th>
              {quarters.map((q) => (
                <th key={q} className="px-2 py-2 text-center font-medium">
                  {q}
                </th>
              ))}
              <th className="px-2 py-2 text-right font-medium">Annual</th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const yqs = quarters
                .map((q) => grid.get(`${year}-${q}`))
                .filter(Boolean) as QuarterlyDetail[]

              let annual = 1
              for (const q of yqs) annual *= 1 + q.cum_return
              annual -= 1

              return (
                <tr key={year} className="border-b border-border/50 last:border-0">
                  <td className="py-2 pr-3 font-medium tabular-nums">{year}</td>
                  {quarters.map((q) => {
                    const d = grid.get(`${year}-${q}`)
                    if (!d) {
                      return (
                        <td key={q} className="px-2 py-2 text-center text-muted-foreground">
                          —
                        </td>
                      )
                    }
                    const pct = d.cum_return * 100
                    return (
                      <td key={q} className="px-2 py-2 text-center">
                        <span
                          className="inline-block min-w-[4.5rem] rounded-md px-2 py-1 text-xs font-medium tabular-nums"
                          style={{ backgroundColor: heatCellBg(pct) }}
                          title={[
                            `Return: ${pct.toFixed(2)}%`,
                            `Turnover: ${d.turnover_pct.toFixed(1)}%`,
                            `Cost: ${d.cost_bps.toFixed(1)} bps`,
                            `Sharpe: ${d.sharpe.toFixed(2)}`,
                            `Max DD: ${(d.max_dd * 100).toFixed(1)}%`,
                            `Positions: ${d.n_positions}`,
                            `Swapped: ${d.n_swapped}`,
                          ].join('\n')}
                        >
                          {pct >= 0 ? '+' : ''}
                          {pct.toFixed(1)}%
                        </span>
                      </td>
                    )
                  })}
                  <td className="px-2 py-2 text-right">
                    <span
                      className={cn(
                        'text-xs font-semibold tabular-nums',
                        annual >= 0 ? 'text-emerald-600' : 'text-red-600',
                      )}
                    >
                      {annual >= 0 ? '+' : ''}
                      {(annual * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Loading / Error / Empty states
// ---------------------------------------------------------------------------

function HistorySkeleton() {
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
      <div className="grid gap-6 xl:grid-cols-2">
        {Array.from({ length: 2 }).map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-4 w-36" />
              <Skeleton className="h-3 w-52" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[340px] w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-56" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-48 w-full" />
        </CardContent>
      </Card>
    </div>
  )
}

function HistoryError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="size-6 text-destructive" />
        </div>
        <div>
          <p className="font-medium">Failed to load historical data</p>
          <p className="text-sm text-muted-foreground">
            The model may not be trained yet or the backend is offline.
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

function EmptyState() {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-4 py-14 text-center">
        <div className="flex size-14 items-center justify-center rounded-full bg-primary/10">
          <Info className="size-7 text-primary" />
        </div>
        <div>
          <p className="text-lg font-medium">No backtest data</p>
          <p className="mt-1 max-w-sm text-sm text-muted-foreground">
            Train the model by running{' '}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">
              python robustness_test.py --quarterly --cs-norm --pub-lag 60 --use-cache
            </code>{' '}
            first.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main content orchestrator
// ---------------------------------------------------------------------------

function HistoryContent({
  perf,
  quarterly,
}: {
  perf: HistoryPerformanceResponse
  quarterly: QuarterlyDetail[]
}) {
  const summary = useMemo(() => {
    const years = Object.entries(perf.per_year).sort(([a], [b]) => Number(a) - Number(b))

    let sCum = 1
    let bCum = 1
    let bestYear = { year: '', ret: -Infinity }
    let worstYear = { year: '', ret: Infinity }

    for (const [year, m] of years) {
      const r = m.long_only.cumulative_return
      sCum *= 1 + r
      bCum *= 1 + m.benchmark.cumulative_return
      if (r > bestYear.ret) bestYear = { year, ret: r }
      if (r < worstYear.ret) worstYear = { year, ret: r }
    }

    const n = years.length
    const annualized = n > 0 ? Math.pow(sCum, 1 / n) - 1 : 0
    const avgSharpe =
      n > 0 ? years.reduce((s, [, m]) => s + m.long_only.sharpe_ratio, 0) / n : 0

    return {
      totalReturn: sCum - 1,
      benchReturn: bCum - 1,
      annualized,
      avgSharpe,
      bestYear,
      worstYear,
      totalCostsBps: perf.total_costs_bps,
      n,
    }
  }, [perf])

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Total Return"
          value={`${summary.totalReturn >= 0 ? '+' : ''}${fmtPct(summary.totalReturn)}`}
          subtitle={`${summary.n} years · Benchmark: ${summary.benchReturn >= 0 ? '+' : ''}${fmtPct(summary.benchReturn)}`}
          icon={TrendingUp}
          trend={summary.totalReturn >= summary.benchReturn ? 'up' : 'down'}
        />
        <KpiCard
          title="Annualized"
          value={`${summary.annualized >= 0 ? '+' : ''}${fmtPct(summary.annualized)}`}
          subtitle={`Best: ${summary.bestYear.year} (${summary.bestYear.ret >= 0 ? '+' : ''}${fmtPct(summary.bestYear.ret)})`}
          icon={BarChart3}
          trend={summary.annualized >= 0 ? 'up' : 'down'}
        />
        <KpiCard
          title="Avg Sharpe"
          value={summary.avgSharpe.toFixed(2)}
          subtitle={`Worst: ${summary.worstYear.year} (${summary.worstYear.ret >= 0 ? '+' : ''}${fmtPct(summary.worstYear.ret)})`}
          icon={CalendarRange}
          trend={summary.avgSharpe >= 0.5 ? 'up' : summary.avgSharpe >= 0 ? 'neutral' : 'down'}
        />
        <KpiCard
          title="Total Costs"
          value={`${summary.totalCostsBps.toFixed(0)} bps`}
          subtitle={`~${(summary.totalCostsBps / Math.max(summary.n, 1)).toFixed(1)} bps / year`}
          icon={RefreshCw}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <YearlyComparisonChart data={perf} />
        <CumulativeReturnChart data={perf} />
      </div>

      {quarterly.length > 0 && <QuarterlyHeatmap data={quarterly} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Signal & position history (archived signals / closed positions)
// ---------------------------------------------------------------------------

function fmtShortDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('de-CH')
}

function HistoricalSignalRow({
  row,
  expanded,
  onToggle,
}: {
  row: PortfolioHistorySignal
  expanded: boolean
  onToggle: () => void
}) {
  const n = row.positions.length
  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/40"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <TableCell className="w-8 align-middle">
          {expanded ? (
            <ChevronDown className="size-4 text-muted-foreground" aria-hidden />
          ) : (
            <ChevronRight className="size-4 text-muted-foreground" aria-hidden />
          )}
        </TableCell>
        <TableCell className="font-medium tabular-nums">{row.cutoff_date}</TableCell>
        <TableCell>
          <Badge variant="secondary" className="font-normal">
            {row.status}
          </Badge>
        </TableCell>
        <TableCell className="text-muted-foreground">
          {row.regime_label ?? '—'}
        </TableCell>
        <TableCell className="text-right tabular-nums">{n}</TableCell>
        <TableCell className="text-muted-foreground text-sm">{fmtShortDate(row.created_at)}</TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="hover:bg-transparent">
          <TableCell colSpan={6} className="bg-muted/20 p-0">
            <div className="max-h-[320px] overflow-auto border-t p-3">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Ticker</TableHead>
                    <TableHead className="text-right">Weight</TableHead>
                    <TableHead className="text-right">Shares</TableHead>
                    <TableHead className="text-right">Entry</TableHead>
                    <TableHead className="text-right">Exit</TableHead>
                    <TableHead className="text-right">P&amp;L</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {row.positions.map((p) => (
                    <TableRow key={p.id}>
                      <TableCell className="font-medium">{p.ticker}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {(p.weight * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {p.shares != null ? p.shares : '—'}
                      </TableCell>
                      <TableCell className="text-right text-sm tabular-nums">
                        {p.entry_price != null ? `CHF ${p.entry_price.toFixed(2)}` : '—'}
                        <span className="block text-xs text-muted-foreground">
                          {fmtShortDate(p.entry_date)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right text-sm tabular-nums">
                        {p.exit_price != null ? `CHF ${p.exit_price.toFixed(2)}` : '—'}
                        <span className="block text-xs text-muted-foreground">
                          {fmtShortDate(p.exit_date)}
                        </span>
                      </TableCell>
                      <TableCell
                        className={cn(
                          'text-right tabular-nums text-sm font-medium',
                          p.pnl_pct != null && p.pnl_pct >= 0 && 'text-emerald-600',
                          p.pnl_pct != null && p.pnl_pct < 0 && 'text-red-600',
                        )}
                      >
                        {p.pnl_pct != null ? fmtPct(p.pnl_pct) : '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function HistoricalBlock() {
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const q = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => api.getPortfolioHistory(),
    staleTime: 60_000,
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Archive className="size-4 opacity-70" />
          Signal &amp; position history
        </CardTitle>
        <CardDescription>
          Past signals with stored positions (newest first). Same data as portfolio history; expands for
          per-ticker entries.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {q.isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        )}
        {q.isError && (
          <p className="text-sm text-destructive">
            {q.error instanceof Error ? q.error.message : 'Could not load history.'}
          </p>
        )}
        {!q.isLoading && !q.isError && (q.data?.length ?? 0) === 0 && (
          <p className="text-sm text-muted-foreground">No archived signals yet.</p>
        )}
        {!q.isLoading && !q.isError && q.data && q.data.length > 0 && (
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>Cutoff</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Regime</TableHead>
                  <TableHead className="text-right">Positions</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {q.data.map((row) => (
                  <HistoricalSignalRow
                    key={row.signal_id}
                    row={row}
                    expanded={expandedId === row.signal_id}
                    onToggle={() =>
                      setExpandedId((id) => (id === row.signal_id ? null : row.signal_id))
                    }
                  />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Backtesting Simulator
// ---------------------------------------------------------------------------

const SIM_STRATEGY_CLR = 'oklch(0.60 0.18 145)'
const SIM_BENCHMARK_CLR = 'oklch(0.58 0.11 260)'

function SimulatorKpis({ summary }: { summary: SimulateSummary }) {
  return (
    <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
      <KpiCard
        title="Final Value"
        value={`CHF ${fmtChf(summary.final_value)}`}
        subtitle={`Started with CHF ${fmtChf(summary.initial_capital)}`}
        icon={DollarSign}
        trend={summary.final_value >= summary.initial_capital ? 'up' : 'down'}
      />
      <KpiCard
        title="Total Return"
        value={`${summary.total_return >= 0 ? '+' : ''}${fmtPct(summary.total_return)}`}
        subtitle={`Benchmark: ${summary.benchmark_total_return >= 0 ? '+' : ''}${fmtPct(summary.benchmark_total_return)}`}
        icon={TrendingUp}
        trend={summary.total_return >= summary.benchmark_total_return ? 'up' : 'down'}
      />
      <KpiCard
        title="CAGR"
        value={`${summary.annualized_return >= 0 ? '+' : ''}${fmtPct(summary.annualized_return)}`}
        icon={BarChart3}
        trend={summary.annualized_return >= 0 ? 'up' : 'down'}
      />
      <KpiCard
        title="Sharpe"
        value={summary.sharpe_ratio.toFixed(2)}
        icon={Target}
        trend={summary.sharpe_ratio >= 0.5 ? 'up' : summary.sharpe_ratio >= 0 ? 'neutral' : 'down'}
      />
      <KpiCard
        title="Max Drawdown"
        value={fmtPct(summary.max_drawdown)}
        icon={TrendingDown}
        trend="down"
      />
      <KpiCard
        title="Total Costs"
        value={`CHF ${fmtChf(summary.total_costs)}`}
        subtitle={`${summary.n_trades} trades`}
        icon={RefreshCw}
      />
    </div>
  )
}

function SimulatorChart({ timeline }: { timeline: SimulateTimelinePoint[] }) {
  const { chartData, xTicks } = useMemo(() => {
    if (!timeline.length) return { chartData: [] as SimulateTimelinePoint[], xTicks: [] as string[] }

    const seen = new Set<string>()
    const data: SimulateTimelinePoint[] = []
    for (let i = 0; i < timeline.length; i++) {
      const month = timeline[i].date.substring(0, 7)
      if (!seen.has(month) || i === timeline.length - 1) {
        seen.add(month)
        data.push(timeline[i])
      }
    }

    const ticks: string[] = []
    let lastYear = ''
    for (const pt of data) {
      const year = pt.date.substring(0, 4)
      if (year !== lastYear) {
        ticks.push(pt.date)
        lastYear = year
      }
    }

    return { chartData: data, xTicks: ticks }
  }, [timeline])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="size-4 opacity-70" />
          Portfolio Value Over Time
        </CardTitle>
        <CardDescription>Simulated portfolio value vs. equal-weight benchmark</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="grad-sim-strat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SIM_STRATEGY_CLR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={SIM_STRATEGY_CLR} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="grad-sim-bench" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SIM_BENCHMARK_CLR} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={SIM_BENCHMARK_CLR} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis
                dataKey="date"
                ticks={xTicks}
                tickFormatter={(d: string) => d.substring(0, 4)}
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tickFormatter={fmtChfAxis}
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={56}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <div className="rounded-lg border bg-card px-3 py-2 text-card-foreground shadow-lg">
                      <p className="mb-1 text-sm font-medium">{String(label)}</p>
                      {payload.map((e) => (
                        <p key={String(e.name)} className="text-xs" style={{ color: String(e.color) }}>
                          {e.name}: CHF {fmtChf(Number(e.value))}
                        </p>
                      ))}
                    </div>
                  )
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Area
                type="monotone"
                name="Strategy"
                dataKey="portfolio_value"
                stroke={SIM_STRATEGY_CLR}
                strokeWidth={2}
                fill="url(#grad-sim-strat)"
              />
              <Area
                type="monotone"
                name="Benchmark"
                dataKey="benchmark_value"
                stroke={SIM_BENCHMARK_CLR}
                strokeWidth={2}
                fill="url(#grad-sim-bench)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

function SimulatorTransactions({ transactions }: { transactions: SimulateTransaction[] }) {
  if (transactions.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ArrowDownUp className="size-4 opacity-70" />
          Transaction History
        </CardTitle>
        <CardDescription>
          {transactions.length} trades executed during the simulation
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-[420px] overflow-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Ticker</TableHead>
                <TableHead>Action</TableHead>
                <TableHead className="text-right">Shares</TableHead>
                <TableHead className="text-right">Price</TableHead>
                <TableHead className="text-right">Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {transactions.map((tx, i) => (
                <TableRow key={i}>
                  <TableCell className="tabular-nums">{tx.date}</TableCell>
                  <TableCell className="font-medium">{tx.ticker}</TableCell>
                  <TableCell>
                    <Badge
                      variant={tx.action === 'buy' ? 'default' : 'destructive'}
                      className="capitalize"
                    >
                      {tx.action}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{tx.shares}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    CHF {tx.price.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    CHF {fmtChf(tx.value)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}

function SimulatorResults({ data }: { data: SimulateResponse }) {
  return (
    <div className="space-y-6">
      <SimulatorKpis summary={data.summary} />
      <SimulatorChart timeline={data.timeline} />
      <SimulatorTransactions transactions={data.transactions} />
    </div>
  )
}

function BacktestSimulator() {
  const [startYear, setStartYear] = useState(2015)
  const [capital, setCapital] = useState(100_000)
  const [costsBps, setCostsBps] = useState(40)

  const mutation = useMutation({
    mutationFn: (body: { start_date: string; initial_capital: number; costs_bps: number }) =>
      api.simulateBacktest(body),
  })

  const handleRun = () => {
    mutation.mutate({
      start_date: `${startYear}-01-01`,
      initial_capital: capital,
      costs_bps: costsBps,
    })
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="size-4 opacity-70" />
            Backtesting Simulator
          </CardTitle>
          <CardDescription>
            Simulate portfolio performance with custom starting capital and date range
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Start Year</label>
              <select
                value={startYear}
                onChange={(e) => setStartYear(Number(e.target.value))}
                className="flex h-9 w-28 rounded-md border border-input bg-transparent px-3 py-1 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                {Array.from({ length: 11 }, (_, i) => 2015 + i).map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Capital (CHF)</label>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                min={1000}
                max={100_000_000}
                step={10_000}
                className="flex h-9 w-36 rounded-md border border-input bg-transparent px-3 py-1 text-sm tabular-nums focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Costs (bps)</label>
              <input
                type="number"
                value={costsBps}
                onChange={(e) => setCostsBps(Number(e.target.value))}
                min={0}
                max={500}
                step={5}
                className="flex h-9 w-24 rounded-md border border-input bg-transparent px-3 py-1 text-sm tabular-nums focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <Button onClick={handleRun} disabled={mutation.isPending}>
              {mutation.isPending ? (
                <>
                  <RefreshCw className="size-3.5 animate-spin" />
                  Running…
                </>
              ) : (
                <>
                  <Play className="size-3.5" />
                  Run Simulation
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {mutation.isPending && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
            <RefreshCw className="size-6 animate-spin text-muted-foreground" />
            <div>
              <p className="font-medium">Running simulation…</p>
              <p className="text-sm text-muted-foreground">
                Building features and simulating trades — this may take a few minutes.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {mutation.isError && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
            <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
              <AlertTriangle className="size-6 text-destructive" />
            </div>
            <div>
              <p className="font-medium">Simulation failed</p>
              <p className="text-sm text-muted-foreground">
                {mutation.error instanceof Error ? mutation.error.message : 'An unexpected error occurred.'}
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={handleRun}>
              <RefreshCw className="size-3.5" />
              Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {mutation.data && <SimulatorResults data={mutation.data} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page export
// ---------------------------------------------------------------------------

export function HistoryPage() {
  const perfQ = useQuery({
    queryKey: ['history', 'performance'],
    queryFn: () => api.getPerformance(),
    staleTime: Infinity,
    gcTime: Infinity,
  })

  const quarterlyQ = useQuery({
    queryKey: ['history', 'quarterly'],
    queryFn: () => api.getQuarterly(),
    staleTime: Infinity,
    gcTime: Infinity,
  })

  const isLoading = perfQ.isLoading || quarterlyQ.isLoading
  const isError = !isLoading && (perfQ.isError || quarterlyQ.isError)
  const perfData = perfQ.data
  const hasYears = perfData != null && Object.keys(perfData.per_year).length > 0

  const handleRefresh = () => {
    perfQ.refetch()
    quarterlyQ.refetch()
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Historical Performance</h1>
          <p className="text-sm text-muted-foreground">
            Walk-forward backtest results and quarterly breakdown
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isLoading}>
          <RefreshCw className={cn('size-3.5', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {isLoading && <HistorySkeleton />}
      {isError && <HistoryError onRetry={handleRefresh} />}
      {!isLoading && perfData != null && !hasYears && <EmptyState />}
      {hasYears && (
        <HistoryContent
          perf={perfData}
          quarterly={quarterlyQ.data?.quarterly_detail ?? []}
        />
      )}

      <HistoricalBlock />

      <BacktestSimulator />
    </div>
  )
}
