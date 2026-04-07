import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { format, formatDistanceToNow } from 'date-fns'
import {
  AlertTriangle,
  Bell,
  CalendarClock,
  Check,
  Info,
  RefreshCw,
  TrendingUp,
} from 'lucide-react'
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type AlertOut } from '@/lib/api'
import { cn } from '@/lib/utils'

function alertTypeIcon(type: string) {
  switch (type) {
    case 'rebalancing_due':
      return <CalendarClock className="size-4 text-blue-500" />
    case 'signal_generated':
      return <TrendingUp className="size-4 text-emerald-500" />
    default:
      return <Bell className="size-4 text-muted-foreground" />
  }
}

function alertTypeLabel(type: string) {
  const labels: Record<string, string> = {
    rebalancing_due: 'Rebalancing',
    signal_generated: 'Signal',
  }
  return labels[type] ?? type
}

function fmtWhen(iso: string | null) {
  if (!iso) return null
  try {
    const d = new Date(iso)
    return {
      relative: formatDistanceToNow(d, { addSuffix: true }),
      absolute: format(d, 'dd.MM.yyyy HH:mm'),
    }
  } catch {
    return null
  }
}

function AlertsSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="flex gap-4 py-4">
            <Skeleton className="size-10 shrink-0 rounded-md" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-3/4 max-w-md" />
              <Skeleton className="h-3 w-32" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function AlertsError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="size-6 text-destructive" />
        </div>
        <div>
          <p className="font-medium">Failed to load alerts</p>
          <p className="text-sm text-muted-foreground">
            Check that the API is running (e.g. port 8000).
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

function AlertRow({
  alert,
  onMarkRead,
  isPending,
}: {
  alert: AlertOut
  onMarkRead: (id: number) => void
  isPending: boolean
}) {
  const when = fmtWhen(alert.created_at)

  return (
    <Card
      className={cn(
        'transition-colors',
        !alert.read && 'border-primary/30 bg-primary/[0.03]',
      )}
    >
      <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-start sm:gap-4">
        <div
          className={cn(
            'flex size-10 shrink-0 items-center justify-center rounded-md border bg-muted/40',
            !alert.read && 'border-primary/20 bg-primary/5',
          )}
        >
          {alertTypeIcon(alert.type)}
        </div>
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="text-[0.65rem]">
              {alertTypeLabel(alert.type)}
            </Badge>
            {!alert.read && (
              <Badge variant="secondary" className="text-[0.65rem]">
                Unread
              </Badge>
            )}
            {when && (
              <span className="text-[0.7rem] text-muted-foreground" title={when.absolute}>
                {when.relative}
              </span>
            )}
          </div>
          <p
            className={cn(
              'text-sm leading-relaxed',
              !alert.read && 'font-medium text-foreground',
              alert.read && 'text-muted-foreground',
            )}
          >
            {alert.message}
          </p>
        </div>
        {!alert.read && (
          <Button
            variant="outline"
            size="sm"
            className="shrink-0 self-start sm:self-center"
            disabled={isPending}
            onClick={() => onMarkRead(alert.id)}
          >
            <Check className="size-3.5" />
            Mark read
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

export function AlertsPage() {
  const queryClient = useQueryClient()
  const [unreadOnly, setUnreadOnly] = useState(false)

  const { data: alerts, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['alerts', unreadOnly],
    queryFn: () => api.getAlerts(unreadOnly),
  })

  const markRead = useMutation({
    mutationFn: (id: number) => api.markAlertRead(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  const unreadCount = alerts?.filter((a) => !a.read).length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Alerts</h1>
          <p className="text-sm text-muted-foreground">
            Rebalancing reminders and signal notifications.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex rounded-md border bg-muted/30 p-0.5">
            <Button
              type="button"
              variant={!unreadOnly ? 'secondary' : 'ghost'}
              size="sm"
              className="h-8 px-3 text-xs"
              onClick={() => setUnreadOnly(false)}
            >
              All
            </Button>
            <Button
              type="button"
              variant={unreadOnly ? 'secondary' : 'ghost'}
              size="sm"
              className="h-8 px-3 text-xs"
              onClick={() => setUnreadOnly(true)}
            >
              Unread only
            </Button>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCw className={cn('size-3.5', isFetching && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {!unreadOnly && alerts && alerts.length > 0 && (
        <Card size="sm">
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2 text-xs">
              <Info className="size-3.5 opacity-70" />
              {unreadCount === 0
                ? 'All caught up — no unread alerts.'
                : `${unreadCount} unread ${unreadCount === 1 ? 'alert' : 'alerts'}`}
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      {isLoading && <AlertsSkeleton />}
      {isError && <AlertsError onRetry={() => refetch()} />}

      {!isLoading && !isError && alerts && (
        <>
          {alerts.length === 0 ? (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Bell className="size-4 opacity-70" />
                  {unreadOnly ? 'No unread alerts' : 'No alerts yet'}
                </CardTitle>
                <CardDescription>
                  {unreadOnly
                    ? 'Switch to the All tab to see past notifications, or check back after the next scheduled run.'
                    : 'Notifications will appear here when signals are generated or rebalancing is due.'}
                </CardDescription>
              </CardHeader>
            </Card>
          ) : (
            <div className="space-y-3">
              {alerts.map((a) => (
                <AlertRow
                  key={a.id}
                  alert={a}
                  onMarkRead={(id) => markRead.mutate(id)}
                  isPending={markRead.isPending && markRead.variables === a.id}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
