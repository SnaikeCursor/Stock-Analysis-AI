import type { SignalOut } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function fmtScore(v: number) {
  return v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3)
}

export function SignalCard({
  signal,
  previousSignal,
}: {
  signal: SignalOut
  previousSignal?: SignalOut | null
}) {
  const showDiff = previousSignal != null
  const prevMap = new Map(
    previousSignal?.portfolio.map((p) => [p.ticker, p]) ?? [],
  )
  const removed =
    previousSignal?.portfolio.filter(
      (p) => !signal.portfolio.some((n) => n.ticker === p.ticker),
    ) ?? []

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">
              {signal.cutoff_date}
            </span>
            <Badge variant="secondary" className="text-[0.65rem]">
              {signal.status}
            </Badge>
            <Badge variant="outline" className="text-[0.60rem] text-muted-foreground">
              {signal.model_phase ?? '9A-PIT'} · {signal.rebalance_freq ?? 'annual'}
            </Badge>
          </div>
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Ticker</TableHead>
            <TableHead className="text-right">Weight</TableHead>
            <TableHead className="text-right">Pred. Return</TableHead>
            {showDiff && <TableHead className="text-right">Change</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {signal.portfolio.map((p) => {
            const prev = prevMap.get(p.ticker)
            const isNew = showDiff && !prev
            const wDiff = prev ? p.weight - prev.weight : null

            return (
              <TableRow key={p.ticker}>
                <TableCell className="font-medium">{p.ticker}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {pct(p.weight)}
                </TableCell>
                <TableCell className={cn(
                  'text-right tabular-nums',
                  p.predicted_return >= 0 ? 'text-emerald-600' : 'text-red-600',
                )}>
                  {fmtScore(p.predicted_return)}
                </TableCell>
                {showDiff && (
                  <TableCell className="text-right">
                    {isNew ? (
                      <Badge className="bg-emerald-600 text-[0.6rem]">
                        new
                      </Badge>
                    ) : wDiff != null && Math.abs(wDiff) > 0.001 ? (
                      <span
                        className={cn(
                          'text-xs font-medium tabular-nums',
                          wDiff > 0 ? 'text-emerald-600' : 'text-red-600',
                        )}
                      >
                        {wDiff > 0 ? '+' : ''}
                        {(wDiff * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </TableCell>
                )}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>

      {removed.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          <span>Removed:</span>
          {removed.map((p) => (
            <Badge
              key={p.ticker}
              variant="outline"
              className="border-red-500/30 text-[0.6rem] text-red-600"
            >
              {p.ticker}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}
