import { ArrowDown, ArrowUp, Check, Pencil, X } from 'lucide-react'
import { useState } from 'react'

import { cn } from '@/lib/utils'
import type { PnlEntry } from '@/lib/api'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

function fmtPrice(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtWeight(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

function fmtChf(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString('de-CH', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

const inputClass =
  'flex h-7 w-full min-w-0 rounded border border-input bg-transparent px-1.5 py-0.5 text-xs tabular-nums shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'

type EditableFields = {
  shares: string
  entry_price: string
  entry_date: string
}

function EditableRow({
  p,
  onSave,
  onCancel,
  isSaving,
}: {
  p: PnlEntry
  onSave: (positionId: number, fields: { shares?: number; entry_price?: number; entry_date?: string }) => void
  onCancel: () => void
  isSaving: boolean
}) {
  const [fields, setFields] = useState<EditableFields>({
    shares: p.shares != null ? String(p.shares) : '',
    entry_price: p.entry_price != null ? String(p.entry_price) : '',
    entry_date: p.entry_date ?? '',
  })

  const handleSave = () => {
    const update: Record<string, number | string> = {}
    const shares = parseInt(fields.shares, 10)
    if (Number.isFinite(shares) && shares > 0 && shares !== p.shares) update.shares = shares
    const price = parseFloat(fields.entry_price.replace(',', '.'))
    if (Number.isFinite(price) && price > 0 && price !== p.entry_price) update.entry_price = price
    if (fields.entry_date && fields.entry_date !== p.entry_date) update.entry_date = fields.entry_date
    if (Object.keys(update).length > 0) onSave(p.id, update)
    else onCancel()
  }

  return (
    <TableRow className="bg-muted/30">
      <TableCell className="font-medium">{p.ticker}</TableCell>
      <TableCell className="text-right">
        <input
          type="number"
          min={1}
          className={cn(inputClass, 'w-16 text-right')}
          value={fields.shares}
          onChange={(e) => setFields((f) => ({ ...f, shares: e.target.value }))}
        />
      </TableCell>
      <TableCell className="text-right tabular-nums">{fmtWeight(p.weight)}</TableCell>
      <TableCell className="text-right">
        <input
          type="text"
          inputMode="decimal"
          className={cn(inputClass, 'w-20 text-right')}
          value={fields.entry_price}
          onChange={(e) => setFields((f) => ({ ...f, entry_price: e.target.value }))}
        />
      </TableCell>
      <TableCell className="text-right tabular-nums text-muted-foreground">
        {fmtChf(p.entry_total)}
      </TableCell>
      <TableCell className="text-right tabular-nums">{fmtPrice(p.current_price)}</TableCell>
      <TableCell className="text-right tabular-nums">{fmtChf(p.current_value)}</TableCell>
      <TableCell className="text-right">
        <span
          className={cn(
            'inline-flex items-center gap-0.5 tabular-nums font-medium',
            p.pnl_pct != null && p.pnl_pct > 0 && 'text-emerald-600',
            p.pnl_pct != null && p.pnl_pct < 0 && 'text-red-600',
          )}
        >
          {fmtPct(p.pnl_pct)}
        </span>
      </TableCell>
      <TableCell className="text-right">
        <span
          className={cn(
            'tabular-nums text-xs font-medium',
            p.pnl_abs != null && p.pnl_abs > 0 && 'text-emerald-600',
            p.pnl_abs != null && p.pnl_abs < 0 && 'text-red-600',
          )}
        >
          {p.pnl_abs != null ? `${p.pnl_abs >= 0 ? '+' : ''}${fmtChf(p.pnl_abs)}` : '—'}
        </span>
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-1">
          <button
            type="button"
            disabled={isSaving}
            onClick={handleSave}
            className="rounded p-0.5 text-emerald-600 hover:bg-emerald-50 disabled:opacity-50"
          >
            <Check className="size-3.5" />
          </button>
          <button
            type="button"
            disabled={isSaving}
            onClick={onCancel}
            className="rounded p-0.5 text-muted-foreground hover:bg-muted"
          >
            <X className="size-3.5" />
          </button>
        </div>
      </TableCell>
    </TableRow>
  )
}

export function PortfolioTable({
  positions,
  onClosePosition,
  onEditPosition,
  editingId,
  onCancelEdit,
  isSaving,
}: {
  positions: PnlEntry[]
  onClosePosition?: (positionId: number) => void
  onEditPosition?: (positionId: number, fields: { shares?: number; entry_price?: number; entry_date?: string }) => void
  editingId?: number | null
  onCancelEdit?: () => void
  isSaving?: boolean
}) {
  if (positions.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
        No active positions. Generate a signal to get started.
      </div>
    )
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Ticker</TableHead>
          <TableHead className="text-right">Shares</TableHead>
          <TableHead className="text-right">Weight</TableHead>
          <TableHead className="text-right">Entry</TableHead>
          <TableHead className="text-right">Invested</TableHead>
          <TableHead className="text-right">Current</TableHead>
          <TableHead className="text-right">Value</TableHead>
          <TableHead className="text-right">P&L %</TableHead>
          <TableHead className="text-right">P&L CHF</TableHead>
          <TableHead className="w-[1%]"> </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {positions.map((p) => {
          if (editingId === p.id && onEditPosition && onCancelEdit) {
            return (
              <EditableRow
                key={p.id}
                p={p}
                onSave={onEditPosition}
                onCancel={onCancelEdit}
                isSaving={isSaving ?? false}
              />
            )
          }

          const isPositive = p.pnl_pct != null && p.pnl_pct > 0
          const isNegative = p.pnl_pct != null && p.pnl_pct < 0

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
              <TableCell className="text-right tabular-nums text-muted-foreground">
                {fmtChf(p.entry_total)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{fmtPrice(p.current_price)}</TableCell>
              <TableCell className="text-right tabular-nums font-medium">
                {fmtChf(p.current_value)}
              </TableCell>
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
                  {fmtPct(p.pnl_pct)}
                </span>
              </TableCell>
              <TableCell className="text-right">
                <span
                  className={cn(
                    'tabular-nums text-xs font-medium',
                    p.pnl_abs != null && p.pnl_abs > 0 && 'text-emerald-600',
                    p.pnl_abs != null && p.pnl_abs < 0 && 'text-red-600',
                  )}
                >
                  {p.pnl_abs != null ? `${p.pnl_abs >= 0 ? '+' : ''}${fmtChf(p.pnl_abs)}` : '—'}
                </span>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  {onEditPosition && (
                    <button
                      type="button"
                      className="text-muted-foreground hover:text-foreground"
                      onClick={() => onEditPosition(p.id, {})}
                    >
                      <Pencil className="size-3" />
                    </button>
                  )}
                  {onClosePosition && (
                    <button
                      type="button"
                      className="text-xs font-medium text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
                      onClick={() => onClosePosition(p.id)}
                    >
                      Close
                    </button>
                  )}
                </div>
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}
