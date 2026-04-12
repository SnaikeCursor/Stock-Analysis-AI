import { useQuery } from '@tanstack/react-query'
import {
  ArrowDown,
  ArrowUp,
  Loader2,
  TrendingUp,
  Wallet,
} from 'lucide-react'

import { api, type MyPortfolioOverview, type PortfolioPerformance } from '@/lib/api'
import { cn } from '@/lib/utils'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

function fmtChf(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

// ---------------------------------------------------------------------------
// KPI Card
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
            trend === 'down' && 'text-red-500',
          )}
        >
          {value}
        </CardTitle>
        {subtitle && (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        )}
      </CardHeader>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Dashboard Page (read-only)
// ---------------------------------------------------------------------------

export function DashboardPage() {
  const perfQuery = useQuery({
    queryKey: ['my-portfolio-performance'],
    queryFn: () => api.getMyPortfolioPerformance(),
    refetchInterval: 60_000,
  })

  const overviewQuery = useQuery({
    queryKey: ['my-portfolio-overview'],
    queryFn: () => api.getMyPortfolio(),
    refetchInterval: 60_000,
  })

  const perf: PortfolioPerformance | undefined = perfQuery.data
  const overview: MyPortfolioOverview | undefined = overviewQuery.data

  const isLoading = perfQuery.isLoading || overviewQuery.isLoading
  const isError = perfQuery.isError && overviewQuery.isError

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="size-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center py-32">
        <p className="text-sm text-muted-foreground">
          Fehler beim Laden der Portfolio-Daten.
        </p>
      </div>
    )
  }

  const positions = overview?.open_positions ?? []

  const returnTrend: 'up' | 'down' | 'neutral' =
    perf?.total_return_pct != null
      ? perf.total_return_pct > 0
        ? 'up'
        : perf.total_return_pct < 0
          ? 'down'
          : 'neutral'
      : 'neutral'

  const pnlTrend: 'up' | 'down' | 'neutral' =
    perf?.total_pnl_abs != null
      ? perf.total_pnl_abs > 0
        ? 'up'
        : perf.total_pnl_abs < 0
          ? 'down'
          : 'neutral'
      : 'neutral'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
        <p className="text-sm text-muted-foreground">
          Aktuelle Performance deines Portfolios (read-only).
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Portfoliowert"
          value={`${fmtChf(perf?.current_value)} CHF`}
          icon={Wallet}
        />
        <KpiCard
          title="Rendite"
          value={fmtPct(perf?.total_return_pct)}
          subtitle={`P&L: ${fmtChf(perf?.total_pnl_abs)} CHF`}
          icon={TrendingUp}
          trend={returnTrend}
        />
        <KpiCard
          title="Investiert (netto)"
          value={`${fmtChf(perf?.net_invested)} CHF`}
          subtitle={`Einzahlungen: ${fmtChf(perf?.total_deposited)} / Auszahlungen: ${fmtChf(perf?.total_withdrawn)}`}
          icon={Wallet}
        />
        <KpiCard
          title="Gewinn / Verlust"
          value={`${fmtChf(perf?.total_pnl_abs)} CHF`}
          icon={pnlTrend === 'up' ? ArrowUp : pnlTrend === 'down' ? ArrowDown : TrendingUp}
          trend={pnlTrend}
        />
      </div>

      {/* Cash & Market Value Detail */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Card size="sm">
          <CardHeader>
            <CardDescription>Cash</CardDescription>
            <CardTitle className="text-lg tabular-nums">
              {fmtChf(perf?.cash_balance)} CHF
            </CardTitle>
          </CardHeader>
        </Card>
        <Card size="sm">
          <CardHeader>
            <CardDescription>Offene Positionen (Marktwert)</CardDescription>
            <CardTitle className="text-lg tabular-nums">
              {fmtChf(perf?.open_market_value)} CHF
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Positions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Offene Positionen</CardTitle>
          <CardDescription>
            {positions.length === 0
              ? 'Keine offenen Positionen vorhanden.'
              : `${positions.length} offene Position${positions.length === 1 ? '' : 'en'}`}
          </CardDescription>
        </CardHeader>
        {positions.length > 0 && (
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Ticker</TableHead>
                  <TableHead className="text-right">Stück</TableHead>
                  <TableHead className="text-right">Kaufpreis</TableHead>
                  <TableHead className="text-right">Aktueller Kurs</TableHead>
                  <TableHead className="text-right">Marktwert</TableHead>
                  <TableHead className="text-right">P&L</TableHead>
                  <TableHead className="text-right">P&L %</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((pos) => {
                  const pnlUp = pos.pnl_pct != null && pos.pnl_pct > 0
                  const pnlDown = pos.pnl_pct != null && pos.pnl_pct < 0
                  return (
                    <TableRow key={pos.id}>
                      <TableCell className="font-medium">{pos.ticker}</TableCell>
                      <TableCell className="text-right tabular-nums">{pos.shares}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtChf(pos.entry_price)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtChf(pos.current_price)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtChf(pos.current_value)}</TableCell>
                      <TableCell
                        className={cn(
                          'text-right tabular-nums',
                          pnlUp && 'text-emerald-600',
                          pnlDown && 'text-red-500',
                        )}
                      >
                        {fmtChf(pos.pnl_abs)}
                      </TableCell>
                      <TableCell
                        className={cn(
                          'text-right tabular-nums',
                          pnlUp && 'text-emerald-600',
                          pnlDown && 'text-red-500',
                        )}
                      >
                        {fmtPct(pos.pnl_pct)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        )}
      </Card>
    </div>
  )
}
