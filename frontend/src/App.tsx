import { useEffect, useRef, useState, type ReactNode } from 'react'
import { AlertCircle, ChevronLeft, ChevronRight, Moon, Sun } from 'lucide-react'
import { api, type Config, type OptimizeRequest, type OptimizeResponse } from './api'
import Overview from './pages/Overview'
import Plan from './pages/Plan'
import Details from './pages/Details'
import SettingsPage from './pages/SettingsPage'
import { Select, cx } from './components/ui'

type Page = 'overview' | 'plan' | 'details' | 'settings'
const PAGES: { id: Page; label: string; short: string; hash: string; title: string }[] = [
  { id: 'overview', label: 'Overview', short: 'Overview', hash: '#/', title: 'Your savings this week' },
  { id: 'plan', label: "Today's plan", short: 'Plan', hash: '#/plan', title: "Today's battery plan" },
  { id: 'details', label: 'Details', short: 'Details', hash: '#/details', title: 'Details' },
  { id: 'settings', label: 'Settings', short: 'Settings', hash: '#/settings', title: 'Settings' },
]
const STORE = 'battery-settings-v1'

const pageFromHash = (): Page => PAGES.find((p) => p.hash === window.location.hash)?.id ?? 'overview'

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
    try { localStorage.setItem('theme', dark ? 'dark' : 'light') } catch { /* private mode */ }
  }, [dark])
  return [dark, setDark] as const
}

/** Saved settings, merged over the defaults and checked against what the server offers. */
function loadSettings(config: Config): OptimizeRequest | null {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? 'null')
    if (!saved) return null
    const v = { ...config.defaults, ...saved, tariff: { ...config.defaults.tariff, ...saved.tariff } } as OptimizeRequest
    if (!config.locations.some((l) => l.id === v.location)) v.location = config.defaults.location
    if (!(v.week >= 0 && v.week < config.weeks.length)) v.week = config.defaults.week
    return v
  } catch {
    return null
  }
}

function Wordmark() {
  return (
    <span className="text-[16px] font-semibold tracking-tight">
      <span className="text-accent">Smart Battery</span> <span className="text-ink dark:text-muted">Optimizer</span>
    </span>
  )
}

/** One line with the current setup and the week, shown above every results page. */
function ScenarioBar({ config, value, onChange, data }: {
  config: Config
  value: OptimizeRequest
  onChange: (v: OptimizeRequest) => void
  data: OptimizeResponse | null
}) {
  const loc = config.locations.find((l) => l.id === value.location)
  const chips = [
    `${value.daily_kwh} kWh/day`, `${value.panel_kw} kW panels`, `${value.battery_kwh} kWh battery`,
    value.outage_hours.length ? `${value.outage_hours.length} h load-shedding` : null,
  ].filter(Boolean)
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-surface px-4 py-3 md:flex-row md:items-center">
      <div className="min-w-0 flex-1">
        <div className="text-[15px] font-semibold text-ink">{loc?.name ?? value.location}</div>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[12px] text-muted">
          {chips.map((c, i) => <span key={c}>{i > 0 && '· '}{c}</span>)}
          <a href="#/settings" className="ml-1 font-medium text-accent hover:underline">Change</a>
        </div>
      </div>
      <div className="flex items-center gap-1.5" aria-label="Week">
        <button type="button" aria-label="Previous week" disabled={value.week === 0} onClick={() => onChange({ ...value, week: value.week - 1 })}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2 disabled:opacity-40">
          <ChevronLeft className="size-4" />
        </button>
        <div className="w-full md:w-[190px]">
          <Select id="week" value={value.week} onChange={(v) => onChange({ ...value, week: Number(v) })}>
            {config.weeks.map((w) => <option key={w.index} value={w.index}>{w.label} {w.start.slice(0, 4)}</option>)}
          </Select>
        </div>
        <button type="button" aria-label="Next week" disabled={value.week === config.weeks.length - 1}
          onClick={() => onChange({ ...value, week: value.week + 1 })}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2 disabled:opacity-40">
          <ChevronRight className="size-4" />
        </button>
      </div>
      {data && <span className="sr-only">Showing {data.meta.period}</span>}
    </div>
  )
}

function Skeleton() {
  return (
    <div className="space-y-4" aria-hidden>
      <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr_1fr]">
        <div className="h-[184px] animate-pulse rounded-xl bg-surface-2 lg:row-span-2" />
        {[0, 1, 2, 3].map((i) => <div key={i} className="h-[86px] animate-pulse rounded-xl bg-surface-2" />)}
      </div>
      <div className="h-[280px] animate-pulse rounded-xl bg-surface-2" />
    </div>
  )
}

function Shell({ children, page, dark, setDark }: { children: ReactNode; page: Page; dark: boolean; setDark: (d: boolean) => void }) {
  return (
    <div className="min-h-screen pb-20 sm:pb-0">
      <header className="sticky top-0 z-20 border-b border-line bg-surface">
        <div className="mx-auto flex h-[60px] max-w-[1180px] items-center gap-4 px-4 sm:px-6">
          <a href="#/" className="flex items-baseline gap-3">
            <Wordmark />
            <span className="hidden text-[12px] text-muted lg:inline">Solar + storage scheduling for Pakistani homes</span>
          </a>
          <nav className="ml-auto hidden h-full items-stretch gap-6 sm:flex" aria-label="Pages">
            {PAGES.map((p) => (
              <a key={p.id} href={p.hash} aria-current={page === p.id ? 'page' : undefined}
                className={cx('-mb-px flex items-center whitespace-nowrap border-b-2 text-[14px] font-medium transition',
                  page === p.id ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink')}>
                {p.label}
              </a>
            ))}
          </nav>
          <button type="button" onClick={() => setDark(!dark)} aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
            className="ml-auto grid size-9 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2 sm:ml-0">
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
        </div>
      </header>
      {children}
      {/* phones: tab bar at the bottom */}
      <nav className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t border-line bg-surface sm:hidden" aria-label="Pages">
        {PAGES.map((p) => (
          <a key={p.id} href={p.hash} aria-current={page === p.id ? 'page' : undefined}
            className={cx('border-t-2 py-3 text-center text-[13px] font-medium', page === p.id ? 'border-accent text-ink' : 'border-transparent text-muted')}>
            {p.short}
          </a>
        ))}
      </nav>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash)
  const [dark, setDark] = useTheme()
  const [config, setConfig] = useState<Config | null>(null)
  const [fatal, setFatal] = useState<string | null>(null)
  const [req, setReq] = useState<OptimizeRequest | null>(null)
  const [firstVisit, setFirstVisit] = useState(false)
  const [data, setData] = useState<OptimizeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => { window.scrollTo(0, 0) }, [page])

  useEffect(() => {
    api.config().then((c) => {
      setConfig(c)
      const saved = loadSettings(c)
      setReq(saved ?? c.defaults)
      if (!saved) {                                  // first visit: start by asking about the home
        setFirstVisit(true)
        if (!window.location.hash) window.location.hash = '#/settings'
      }
    }).catch((e) => setFatal(e.message))
  }, [])

  useEffect(() => {
    if (!req) return
    try { localStorage.setItem(STORE, JSON.stringify(req)) } catch { /* private mode */ }
    const id = setTimeout(() => {
      abort.current?.abort()
      const ctl = new AbortController()
      abort.current = ctl
      setLoading(true)
      api.optimize(req, ctl.signal)
        .then((d) => { setData(d); setError(null) })
        .catch((e) => { if (e.name !== 'AbortError') setError(e.message) })
        .finally(() => { if (abort.current === ctl) setLoading(false) })
    }, 350)
    return () => clearTimeout(id)
  }, [req])

  const current = PAGES.find((p) => p.id === page)!
  const body = (() => {
    if (fatal) {
      return (
        <div role="alert" className="mx-auto mt-10 max-w-[640px] rounded-xl border border-line bg-surface p-6">
          <div className="flex items-center gap-2 text-[15px] font-semibold text-bad"><AlertCircle className="size-4" />Service unavailable</div>
          <p className="mt-2 text-[14px] text-ink-2">The planner could not be loaded. Please refresh the page or try again in a moment.</p>
        </div>
      )
    }
    if (!config || !req) return <Skeleton />
    if (page === 'settings') {
      return <SettingsPage config={config} value={req} onChange={setReq} firstVisit={firstVisit} />
    }
    return (
      <div className="space-y-4">
        <div className="flex items-baseline justify-between gap-3">
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{current.title}</h1>
        </div>
        <ScenarioBar config={config} value={req} onChange={setReq} data={data} />
        <div className="relative h-0.5 overflow-hidden rounded-full" aria-hidden>
          {loading && <div className="animate-progress absolute inset-y-0 w-1/3 rounded-full bg-accent" />}
        </div>
        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-xl border border-line bg-surface px-4 py-3 text-[13px] text-bad">
            <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
            <div>
              <span className="font-semibold">Could not run these settings.</span> {error}{' '}
              <a href="#/settings" className="font-medium underline">Open settings</a>
              {data && <span className="text-muted"> Results below are from your last valid settings.</span>}
            </div>
          </div>
        )}
        {!data ? <Skeleton /> : (
          <div className={cx('transition-opacity', loading && 'opacity-60')} aria-busy={loading}>
            {page === 'overview' && <Overview data={data} />}
            {page === 'plan' && <Plan data={data} />}
            {page === 'details' && <Details data={data} />}
          </div>
        )}
      </div>
    )
  })()

  return (
    <Shell page={page} dark={dark} setDark={setDark}>
      <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6">{body}</main>
    </Shell>
  )
}
