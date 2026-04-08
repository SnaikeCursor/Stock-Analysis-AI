import { useCallback, useState } from 'react'
import { useIsFetching, useQueryClient } from '@tanstack/react-query'
import { NavLink, Outlet } from 'react-router-dom'
import {
  Activity,
  Bell,
  Copy,
  History,
  KeyRound,
  LayoutDashboard,
  RefreshCw,
  Wallet,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  getPortfolioDisplayCode,
  getPortfolioUserId,
  setPortfolioUserId,
} from '@/lib/api'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/portfolio', label: 'Mein Portfolio', icon: Wallet, end: false },
  { to: '/history', label: 'History', icon: History, end: false },
  { to: '/alerts', label: 'Alerts', icon: Bell, end: false },
] as const

export function Layout() {
  const queryClient = useQueryClient()
  const isFetching = useIsFetching()
  const [codeOpen, setCodeOpen] = useState(false)
  const [importRaw, setImportRaw] = useState('')
  const [importErr, setImportErr] = useState<string | null>(null)
  const [portfolioKey, setPortfolioKey] = useState(0)

  const displayCode = getPortfolioDisplayCode()

  const copyFullId = useCallback(async () => {
    const id = getPortfolioUserId()
    try {
      await navigator.clipboard.writeText(id)
    } catch {
      /* ignore */
    }
  }, [])

  const applyImport = useCallback(() => {
    setImportErr(null)
    if (setPortfolioUserId(importRaw)) {
      setPortfolioKey((k) => k + 1)
      setCodeOpen(false)
      setImportRaw('')
      void queryClient.invalidateQueries()
    } else {
      setImportErr('Ungueltige UUID. Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')
    }
  }, [importRaw, queryClient])

  return (
    <div className="flex min-h-svh flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border/80 bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
          <div className="flex items-center gap-2">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Activity className="size-5" aria-hidden />
            </span>
            <div className="leading-tight">
              <p className="text-sm font-semibold tracking-tight">Stock Analysis AI</p>
              <p className="text-xs text-muted-foreground">SPI Lag60-SA signals</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <nav className="flex flex-wrap gap-1" aria-label="Main">
              {navItems.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    cn(
                      'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-muted text-foreground shadow-sm'
                        : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                    )
                  }
                >
                  <Icon className="size-3.5 opacity-80" aria-hidden />
                  {label}
                </NavLink>
              ))}
            </nav>
            <div
              key={portfolioKey}
              className="flex flex-wrap items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2 py-1 text-xs"
            >
              <span className="text-muted-foreground">Portfolio</span>
              <code className="rounded bg-background px-1.5 py-0.5 font-mono text-[0.7rem]">{displayCode}</code>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 px-1.5"
                onClick={() => void copyFullId()}
                title="Volle UUID kopieren"
              >
                <Copy className="size-3.5" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 px-1.5"
                onClick={() => setCodeOpen(true)}
                title="Anderes Geraet: UUID einfuegen"
              >
                <KeyRound className="size-3.5" />
              </Button>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="size-8 shrink-0"
              onClick={() => queryClient.invalidateQueries()}
              aria-label="Refresh all data"
            >
              <RefreshCw className={cn('size-4', isFetching > 0 && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </header>
      {codeOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="portfolio-import-title"
        >
          <Card className="w-full max-w-md shadow-lg">
            <CardHeader>
              <CardTitle id="portfolio-import-title">Portfolio-ID</CardTitle>
              <CardDescription>
                Volle UUID von einem anderen Geraet einfuegen, um dasselbe Portfolio zu laden. Die UUID
                erhaeltst du ueber &quot;Kopieren&quot; neben dem Kurz-Code.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                type="text"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm font-mono"
                placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                value={importRaw}
                onChange={(e) => setImportRaw(e.target.value)}
              />
              {importErr && <p className="text-sm text-destructive">{importErr}</p>}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" size="sm" onClick={() => setCodeOpen(false)}>
                  Abbrechen
                </Button>
                <Button type="button" size="sm" onClick={applyImport}>
                  Uebernehmen
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
