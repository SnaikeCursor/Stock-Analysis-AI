import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
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
  BarChart3,
  CalendarRange,
  Info,
  RefreshCw,
  TrendingUp,
} from 'lucide-react'

import { api } from '@/lib/api'
import type { HistoryPerformanceResponse, QuarterlyDetail } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

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
    </div>
  )
}
