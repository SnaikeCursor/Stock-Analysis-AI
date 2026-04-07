import { useIsFetching, useQueryClient } from '@tanstack/react-query'
import { NavLink, Outlet } from 'react-router-dom'
import { Activity, Bell, History, LayoutDashboard, RefreshCw } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/history', label: 'History', icon: History, end: false },
  { to: '/alerts', label: 'Alerts', icon: Bell, end: false },
] as const

export function Layout() {
  const queryClient = useQueryClient()
  const isFetching = useIsFetching()

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
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
