import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Bell,
  CalendarClock,
  Info,
  RefreshCw,
  TrendingUp,
  Wallet,
} from 'lucide-react'

import { api } from '@/lib/api'
import type { DashboardPosition, DashboardResponse, PnlEntry } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { PortfolioTable } from '@/components/PortfolioTable'

function computeKpis(positions: DashboardPosition[]) {
  if (positions.length === 0) return null

  const withPnl = positions.filter((p) => p.pnl_pct != null)
  if (withPnl.length === 0) return null

  const totalWeight = withPnl.reduce((s, p) => s + p.weight, 0)
  const weightedReturn = totalWeight > 0
    ? withPnl.reduce((s, p) => s + p.weight * (p.pnl_pct ?? 0), 0) / totalWeight
    : 0

  const sorted = [...withPnl].sort((a, b) => (b.pnl_pct ?? 0) - (a.pnl_pct ?? 0))
  const best = sorted[0]
  const worst = sorted[sorted.length - 1]

  return { weightedReturn, best, worst, count: positions.length }
}

function alertTypeIcon(type: string) {
  switch (type) {
    case 'rebalancing_due':
      return <CalendarClock className="size-3.5 text-blue-500" />
    case 'signal_generated':
      return <TrendingUp className="size-3.5 text-emerald-500" />
    default:
      return <Bell className="size-3.5 text-muted-foreground" />
  }
}

function alertTypeBadge(type: string) {
  const labels: Record<string, string> = {
    rebalancing_due: 'Rebalancing',
    signal_generated: 'Signal',
  }
  return labels[type] ?? type
}

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
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <Skeleton className="h-4 w-32" />
          </CardHeader>
          <CardContent className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton className="h-8 w-full" key={i} />
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-24" />
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-20 w-full" />
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

function DashboardContent({ data }: { data: DashboardResponse }) {
  const { signal, positions, next_rebalancing, alerts } = data
  const kpis = computeKpis(positions)

  const portfolioReturn = kpis ? kpis.weightedReturn : null
  const returnTrend: 'up' | 'down' | 'neutral' =
    portfolioReturn == null ? 'neutral' : portfolioReturn >= 0 ? 'up' : 'down'

  return (
    <div className="space-y-6">
      {/* KPI row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Portfolio Return"
          value={portfolioReturn != null ? `${portfolioReturn >= 0 ? '+' : ''}${(portfolioReturn * 100).toFixed(2)}%` : '—'}
          subtitle={kpis ? `${kpis.count} active positions` : 'No positions'}
          icon={Wallet}
          trend={returnTrend}
        />

        <KpiCard
          title="Best Position"
          value={kpis?.best ? kpis.best.ticker : '—'}
          subtitle={
            kpis?.best?.pnl_pct != null
              ? `${kpis.best.pnl_pct >= 0 ? '+' : ''}${(kpis.best.pnl_pct * 100).toFixed(2)}%`
              : undefined
          }
          icon={ArrowUp}
          trend={kpis?.best?.pnl_pct != null ? (kpis.best.pnl_pct >= 0 ? 'up' : 'down') : 'neutral'}
        />

        <KpiCard
          title="Worst Position"
          value={kpis?.worst ? kpis.worst.ticker : '—'}
          subtitle={
            kpis?.worst?.pnl_pct != null
              ? `${kpis.worst.pnl_pct >= 0 ? '+' : ''}${(kpis.worst.pnl_pct * 100).toFixed(2)}%`
              : undefined
          }
          icon={ArrowDown}
          trend={kpis?.worst?.pnl_pct != null ? (kpis.worst.pnl_pct >= 0 ? 'up' : 'down') : 'neutral'}
        />

        <KpiCard
          title="Next Rebalancing"
          value={`${next_rebalancing.days_until}d`}
          subtitle={new Date(next_rebalancing.date).toLocaleDateString('de-CH', {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
          })}
          icon={CalendarClock}
        />
      </div>

      {/* Main content: Portfolio + Alerts */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Portfolio table */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wallet className="size-4 opacity-70" />
              Active Portfolio
            </CardTitle>
            {signal && (
              <CardDescription>
                Signal from {signal.cutoff_date}
                {' · '}
                <Badge variant="secondary" className="ml-1 text-[0.65rem]">
                  {signal.status}
                </Badge>
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            <PortfolioTable
              positions={positions.map((p): PnlEntry => ({
                ...p,
                shares: null,
                entry_total: null,
                current_value: null,
                pnl_abs: null,
              }))}
            />
          </CardContent>
        </Card>

        {/* Sidebar: Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="size-4 opacity-70" />
              Recent Alerts
              {alerts.length > 0 && (
                <Badge variant="secondary" className="ml-auto tabular-nums">
                  {alerts.length}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="size-3.5 opacity-60" />
                No unread alerts
              </div>
            ) : (
              <ul className="space-y-3">
                {alerts.map((a) => (
                  <li key={a.id} className="flex items-start gap-2.5">
                    <span className="mt-0.5">{alertTypeIcon(a.type)}</span>
                    <div className="flex-1 space-y-0.5">
                      <p className="text-sm leading-snug">{a.message}</p>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-[0.6rem]">
                          {alertTypeBadge(a.type)}
                        </Badge>
                        {a.created_at && (
                          <span className="text-[0.65rem] text-muted-foreground">
                            {formatDistanceToNow(new Date(a.created_at), { addSuffix: true })}
                          </span>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function DashboardPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['dashboard'],
    queryFn: api.getDashboard,
    refetchInterval: 60_000,
  })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Lag60-SA model · Semi-annual rebalancing · Portfolio and alerts at a glance
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isLoading}
        >
          <RefreshCw className={cn('size-3.5', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {isLoading && <DashboardSkeleton />}
      {isError && <DashboardError onRetry={() => refetch()} />}
      {data && <DashboardContent data={data} />}
    </div>
  )
}
