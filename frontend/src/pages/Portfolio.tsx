import { useCallback, useMemo, useState, type ElementType } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { format } from 'date-fns'
import {
  AlertTriangle,
  CircleDollarSign,
  Loader2,
  Plus,
  Trash2,
  TrendingUp,
  Wallet,
  X,
} from 'lucide-react'

import {
  api,
  type MyClosedPosition,
  type MyOpenPosition,
  type MyPortfolioSummary,
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

function fmtChf(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

function todayIso() {
  return new Date().toISOString().slice(0, 10)
}

const inputClass =
  'flex h-8 w-full min-w-0 rounded-md border border-input bg-transparent px-2 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'

export function PortfolioPage() {
  const queryClient = useQueryClient()
  const [depositRaw, setDepositRaw] = useState('')
  const [ticker, setTicker] = useState('')
  const [sharesRaw, setSharesRaw] = useState('')
  const [entryPriceRaw, setEntryPriceRaw] = useState('')
  const [entryDate, setEntryDate] = useState(todayIso)
  const [closingId, setClosingId] = useState<number | null>(null)
  const [exitPriceRaw, setExitPriceRaw] = useState('')
  const [exitDate, setExitDate] = useState('')

  const overviewQuery = useQuery({
    queryKey: ['my-portfolio'],
    queryFn: api.getMyPortfolio,
  })
  const summaryQuery = useQuery({
    queryKey: ['my-portfolio-summary'],
    queryFn: api.getMyPortfolioSummary,
  })

  const notionalPreview = useMemo(() => {
    const sh = parseInt(sharesRaw, 10) || 0
    const px = parseFloat(entryPriceRaw.replace(',', '.')) || 0
    return sh > 0 && px > 0 ? sh * px : 0
  }, [sharesRaw, entryPriceRaw])

  const feePreviewQuery = useQuery({
    queryKey: ['swissquote-fee', notionalPreview],
    queryFn: () => api.getSwissquoteFeeEstimate(notionalPreview),
    enabled: notionalPreview > 0,
  })

  const invalidate = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['my-portfolio'] })
    void queryClient.invalidateQueries({ queryKey: ['my-portfolio-summary'] })
  }, [queryClient])

  const depositMut = useMutation({
    mutationFn: api.depositMyPortfolio,
    onSuccess: () => invalidate(),
  })

  const addMut = useMutation({
    mutationFn: api.addMyPosition,
    onSuccess: () => {
      invalidate()
      setTicker('')
      setSharesRaw('')
      setEntryPriceRaw('')
      setEntryDate(todayIso())
    },
  })

  const closeMut = useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: number
      body: { exit_price: number; exit_date?: string | null }
    }) => api.closeMyPosition(id, body),
    onSuccess: () => {
      invalidate()
      setClosingId(null)
      setExitPriceRaw('')
      setExitDate('')
    },
  })

  const deleteMut = useMutation({
    mutationFn: api.deleteMyPosition,
    onSuccess: () => invalidate(),
  })

  const handleDeposit = () => {
    const amount = parseFloat(depositRaw.replace(',', '.'))
    if (!Number.isFinite(amount) || amount <= 0) return
    depositMut.mutate({ amount })
    setDepositRaw('')
  }

  const handleAdd = () => {
    const sh = parseInt(sharesRaw, 10)
    const px = parseFloat(entryPriceRaw.replace(',', '.'))
    if (!ticker.trim() || sh < 1 || !Number.isFinite(px) || px <= 0) return
    addMut.mutate({
      ticker: ticker.trim().toUpperCase(),
      shares: sh,
      entry_price: px,
      entry_date: entryDate,
    })
  }

  const openClose = (row: MyOpenPosition) => {
    setClosingId(row.id)
    setExitPriceRaw(row.current_price != null ? String(row.current_price) : '')
    setExitDate('')
  }

  const submitClose = () => {
    if (closingId == null) return
    const px = parseFloat(exitPriceRaw.replace(',', '.'))
    if (!Number.isFinite(px) || px <= 0) return
    closeMut.mutate({
      id: closingId,
      body: { exit_price: px, exit_date: exitDate.trim() || null },
    })
  }

  const data = overviewQuery.data
  const summary: MyPortfolioSummary | undefined = summaryQuery.data
  const loading = overviewQuery.isLoading || summaryQuery.isLoading
  const err = overviewQuery.error || summaryQuery.error

  const closingRow =
    closingId != null ? data?.open_positions.find((p) => p.id === closingId) : undefined

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Mein Portfolio</h1>
        <p className="text-sm text-muted-foreground">
          Cash, Positionen und Schaetzung Swissquote-Gebuehren (CHF). Identitaet: Kurz-Code in der
          Kopfzeile.
        </p>
      </div>

      {err && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-start gap-2 pt-4 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              {err instanceof Error ? err.message : 'Daten konnten nicht geladen werden.'}
            </span>
          </CardContent>
        </Card>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Laden…
        </div>
      )}

      {!loading && summary && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Kpi
            title="Gesamtwert"
            value={`${fmtChf(summary.total_portfolio_value)} CHF`}
            icon={Wallet}
            subtitle={`Cash ${fmtChf(summary.cash_balance)}`}
          />
          <Kpi
            title="Unrealisierte P&L"
            value={fmtChf(summary.unrealized_pnl)}
            icon={TrendingUp}
            trend={summary.unrealized_pnl >= 0 ? 'up' : 'down'}
          />
          <Kpi
            title="Realisierte P&L"
            value={fmtChf(summary.realized_pnl)}
            icon={TrendingUp}
            trend={summary.realized_pnl >= 0 ? 'up' : 'down'}
          />
          <Kpi
            title="Gebuehren (geschaetzt)"
            value={`${fmtChf(summary.total_fees_paid)} CHF`}
            icon={CircleDollarSign}
            subtitle="Swissquote-Stufe + 0.85%"
          />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cash einzahlen</CardTitle>
            <CardDescription>Betrag in CHF auf das freie Cash-Konto.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap items-end gap-2">
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground" htmlFor="dep">
                Betrag (CHF)
              </label>
              <input
                id="dep"
                type="text"
                inputMode="decimal"
                className={cn(inputClass, 'w-40')}
                value={depositRaw}
                onChange={(e) => setDepositRaw(e.target.value)}
              />
            </div>
            <Button size="sm" onClick={handleDeposit} disabled={depositMut.isPending}>
              {depositMut.isPending ? <Loader2 className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />}
              Einzahlen
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Position hinzufuegen</CardTitle>
            <CardDescription>Kauf: Notional + Swissquote-Gebuehr werden vom Cash abgezogen.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Ticker (z. B. NESN.SW)</label>
                <input
                  className={inputClass}
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value)}
                  placeholder="NESN.SW"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Stueck</label>
                <input
                  type="number"
                  min={1}
                  className={inputClass}
                  value={sharesRaw}
                  onChange={(e) => setSharesRaw(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Einstiegspreis (CHF)</label>
                <input
                  type="text"
                  inputMode="decimal"
                  className={inputClass}
                  value={entryPriceRaw}
                  onChange={(e) => setEntryPriceRaw(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Datum</label>
                <input
                  type="date"
                  className={inputClass}
                  value={entryDate}
                  onChange={(e) => setEntryDate(e.target.value)}
                />
              </div>
            </div>
            {notionalPreview > 0 && feePreviewQuery.data && (
              <p className="text-xs text-muted-foreground">
                Notional ~{fmtChf(notionalPreview)} CHF · Geschaetzte Gebuehr ca.{' '}
                {fmtChf(feePreviewQuery.data.fee_chf)} CHF · Total Abgang ca.{' '}
                {fmtChf(notionalPreview + feePreviewQuery.data.fee_chf)} CHF
              </p>
            )}
            {addMut.isError && (
              <p className="text-sm text-destructive">
                {addMut.error instanceof Error ? addMut.error.message : 'Fehler'}
              </p>
            )}
          </CardContent>
          <CardFooter>
            <Button size="sm" onClick={handleAdd} disabled={addMut.isPending}>
              {addMut.isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
              Position buchen
            </Button>
          </CardFooter>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <TrendingUp className="size-4 opacity-70" />
            Offene Positionen
          </CardTitle>
          <CardDescription>Live-Kurs aus Backend-Cache (yfinance), wo verfuegbar.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <OpenPositionsTable
            rows={data?.open_positions ?? []}
            onClose={openClose}
            onDelete={(id) => {
              if (confirm('Offene Position wirklich loeschen? Einstiegsnotional wird gutgeschrieben (Gebuehr bleibt verbucht).')) {
                deleteMut.mutate(id)
              }
            }}
            deleting={deleteMut.isPending}
          />
        </CardContent>
      </Card>

      {closingId != null && closingRow && (
        <Card className="border-amber-200 bg-amber-50/40 dark:border-amber-900 dark:bg-amber-950/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Position schliessen: {closingRow.ticker}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Exit-Preis</label>
              <input
                className={inputClass}
                value={exitPriceRaw}
                onChange={(e) => setExitPriceRaw(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Exit-Datum</label>
              <input
                type="date"
                className={inputClass}
                value={exitDate}
                onChange={(e) => setExitDate(e.target.value)}
              />
            </div>
            <Button size="sm" onClick={submitClose} disabled={closeMut.isPending}>
              {closeMut.isPending ? '…' : 'Schliessen'}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setClosingId(null)}>
              <X className="size-3.5" />
            </Button>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Geschlossene Positionen</CardTitle>
          <CardDescription>Historie; loeschen macht die Buchung rueckgaengig (Cash + P&L).</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <ClosedPositionsTable
            rows={data?.closed_positions ?? []}
            onDelete={(id) => {
              if (confirm('Geschlossene Position aus Historie entfernen?')) {
                deleteMut.mutate(id)
              }
            }}
            deleting={deleteMut.isPending}
          />
        </CardContent>
      </Card>
    </div>
  )
}

function Kpi({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
}: {
  title: string
  value: string
  subtitle?: string
  icon: ElementType
  trend?: 'up' | 'down'
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
            'text-lg tabular-nums',
            trend === 'up' && 'text-emerald-600',
            trend === 'down' && 'text-red-600',
          )}
        >
          {value}
        </CardTitle>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </CardHeader>
    </Card>
  )
}

function OpenPositionsTable({
  rows,
  onClose,
  onDelete,
  deleting,
}: {
  rows: MyOpenPosition[]
  onClose: (p: MyOpenPosition) => void
  onDelete: (id: number) => void
  deleting: boolean
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">Keine offenen Positionen.</p>
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Ticker</TableHead>
          <TableHead className="text-right">Stueck</TableHead>
          <TableHead className="text-right">Einstieg</TableHead>
          <TableHead>Datum</TableHead>
          <TableHead className="text-right">Gebuehr</TableHead>
          <TableHead className="text-right">Kurs</TableHead>
          <TableHead className="text-right">P&L</TableHead>
          <TableHead className="text-right">Aktion</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((p) => (
          <TableRow key={p.id}>
            <TableCell className="font-mono font-medium">{p.ticker}</TableCell>
            <TableCell className="text-right tabular-nums">{p.shares}</TableCell>
            <TableCell className="text-right tabular-nums">{fmtChf(p.entry_price)}</TableCell>
            <TableCell className="text-muted-foreground">{formatDate(p.entry_date)}</TableCell>
            <TableCell className="text-right tabular-nums">{fmtChf(p.entry_fee)}</TableCell>
            <TableCell className="text-right tabular-nums">{fmtChf(p.current_price)}</TableCell>
            <TableCell className="text-right tabular-nums">
              <span className={cn(p.pnl_abs != null && p.pnl_abs < 0 && 'text-red-600')}>
                {fmtChf(p.pnl_abs)} ({fmtPct(p.pnl_pct)})
              </span>
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-1">
                <Button type="button" size="sm" variant="outline" onClick={() => onClose(p)}>
                  Schliessen
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={deleting}
                  onClick={() => onDelete(p.id)}
                  aria-label="Loeschen"
                >
                  <Trash2 className="size-3.5 text-destructive" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function ClosedPositionsTable({
  rows,
  onDelete,
  deleting,
}: {
  rows: MyClosedPosition[]
  onDelete: (id: number) => void
  deleting: boolean
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">Noch keine geschlossenen Positionen.</p>
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Ticker</TableHead>
          <TableHead className="text-right">P&L CHF</TableHead>
          <TableHead className="text-right">P&L %</TableHead>
          <TableHead>Exit</TableHead>
          <TableHead className="text-right" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((p) => (
          <TableRow key={p.id}>
            <TableCell className="font-mono">{p.ticker}</TableCell>
            <TableCell className="text-right tabular-nums">
              <Badge variant={p.pnl_abs != null && p.pnl_abs >= 0 ? 'secondary' : 'destructive'}>
                {fmtChf(p.pnl_abs)}
              </Badge>
            </TableCell>
            <TableCell className="text-right tabular-nums">{fmtPct(p.pnl_pct)}</TableCell>
            <TableCell className="text-muted-foreground text-xs">
              {formatDate(p.exit_date)} @ {fmtChf(p.exit_price)}
            </TableCell>
            <TableCell className="text-right">
              <Button
                type="button"
                size="sm"
                variant="ghost"
                disabled={deleting}
                onClick={() => onDelete(p.id)}
              >
                <Trash2 className="size-3.5" />
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return format(new Date(iso), 'dd.MM.yyyy')
  } catch {
    return iso
  }
}
