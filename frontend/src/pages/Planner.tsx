import { useEffect, useRef, useState } from 'react'
import { AlertCircle, SlidersHorizontal, X } from 'lucide-react'
import { api, type Config, type OptimizeRequest, type OptimizeResponse, type StrategyId } from '../api'
import Controls from '../components/Controls'
import HourlyCharts from '../components/HourlyCharts'
import { DailyTable, ScheduleTable, StrategyComparison } from '../components/Results'
import { Stat, cx } from '../components/ui'
import { num, pct, pkr } from '../lib/format'

function Hero({ data }: { data: OptimizeResponse }) {
  const s = data.summary
  const mpc = s.mpc!, none = s.none!, self = s.selfuse!
  const saving = none.total_cost - mpc.total_cost
  const vsSelf = self.total_cost - mpc.total_cost
  const outages = data.request.outage_hours.length > 0
  return (
    <div className="grid gap-3 lg:grid-cols-[1.35fr_1fr_1fr]">
      <div className="rounded-xl border border-line bg-surface px-5 py-4 lg:row-span-2">
        <div className="text-[13px] font-medium text-muted">Smart schedule saves this week, vs no battery</div>
        <div className="mt-1.5 text-[44px] font-semibold leading-none tracking-tight text-ink">{pkr(saving)}</div>
        <div className="mt-2 text-[13px] text-ink-2">
          {none.total_cost > 0 && <>{pct(100 * saving / none.total_cost, 0)} lower bill · </>}≈ {pkr(saving * 52 / 12)} per month at this rate
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-3 text-[13px]">
          <div>
            <div className="text-muted">Without battery</div>
            <div className="tnum font-semibold text-ink">{pkr(none.total_cost)}</div>
          </div>
          <div>
            <div className="text-muted">With smart schedule</div>
            <div className="tnum font-semibold text-ink">{pkr(mpc.total_cost)}</div>
          </div>
        </div>
      </div>
      <Stat label="Versus basic battery use" value={`${pkr(Math.abs(vsSelf))} ${vsSelf >= 0 ? 'less' : 'more'}`}
        sub={vsSelf >= 0.5 ? 'Smart schedule is cheaper' : vsSelf <= -0.5 ? 'Basic use is cheaper this week' : 'About the same'}
        tone={vsSelf >= 0.5 ? 'good' : vsSelf <= -0.5 ? 'bad' : undefined} />
      <Stat label="Self-sufficiency" value={pct(mpc.self_sufficiency_pct, 0)}
        sub={`of demand met without importing (no battery: ${pct(none.self_sufficiency_pct, 0)})`} />
      {outages ? (
        <Stat label="Unmet demand in outages" value={`${num(mpc.unserved_kwh)} kWh`}
          sub={`Basic battery use: ${num(self.unserved_kwh)} kWh · no battery: ${num(none.unserved_kwh)} kWh`}
          tone={mpc.unserved_kwh <= self.unserved_kwh ? 'good' : 'bad'} />
      ) : (
        <Stat label="Grid import" value={`${num(mpc.import_kwh, 0)} kWh`} sub={`No battery: ${num(none.import_kwh, 0)} kWh`} />
      )}
      <Stat label="Battery use" value={`${num(mpc.cycles)} cycles`}
        sub={data.request.deg_cost > 0 ? `Wear ${pkr(mpc.wear_cost)}` : 'full charge–discharge equivalents'} />
    </div>
  )
}

function Skeleton() {
  return (
    <div className="space-y-4" aria-hidden>
      <div className="grid gap-3 lg:grid-cols-[1.35fr_1fr_1fr]">
        <div className="h-[172px] animate-pulse rounded-xl bg-surface-2 lg:row-span-2" />
        {[0, 1, 2, 3].map((i) => <div key={i} className="h-[80px] animate-pulse rounded-xl bg-surface-2" />)}
      </div>
      <div className="h-[300px] animate-pulse rounded-xl bg-surface-2" />
      <div className="h-[640px] animate-pulse rounded-xl bg-surface-2" />
    </div>
  )
}

export default function Planner({ config }: { config: Config }) {
  const [req, setReq] = useState<OptimizeRequest>(config.defaults)
  const [data, setData] = useState<OptimizeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [focus, setFocus] = useState<StrategyId>('mpc')
  const [panelOpen, setPanelOpen] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
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

  // only the options in STRATEGY_ORDER are shown; fall back to the smart schedule
  const shownFocus: StrategyId = data?.hourly.series[focus] ? focus : 'mpc'

  return (
    <div className="mx-auto grid max-w-[1440px] gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <button type="button" onClick={() => setPanelOpen(true)}
        className="flex items-center justify-center gap-2 rounded-lg border border-line bg-surface px-4 py-2.5 text-[14px] font-medium text-ink lg:hidden">
        <SlidersHorizontal className="size-4" aria-hidden />Scenario settings
      </button>

      {panelOpen && <div className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setPanelOpen(false)} aria-hidden />}
      <aside className={cx(
        'z-40 overflow-y-auto border-line bg-surface',
        'fixed inset-y-0 left-0 w-[min(340px,90vw)] border-r transition-transform lg:static lg:w-auto lg:translate-x-0',
        'lg:sticky lg:top-[76px] lg:max-h-[calc(100vh-96px)] lg:rounded-xl lg:border',
        panelOpen ? 'translate-x-0' : '-translate-x-full',
      )}>
        <div className="flex items-center justify-between border-b border-line px-5 py-3 lg:hidden">
          <span className="text-[14px] font-semibold">Scenario settings</span>
          <button type="button" aria-label="Close settings" onClick={() => setPanelOpen(false)} className="rounded-md p-1 hover:bg-surface-2"><X className="size-4" /></button>
        </div>
        <Controls config={config} value={req} onChange={setReq} />
      </aside>

      <main className="min-w-0 space-y-4">
        <div className="relative h-0.5 overflow-hidden rounded-full" aria-hidden>
          {loading && <div className="animate-progress absolute inset-y-0 w-1/3 rounded-full bg-accent" />}
        </div>
        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-xl border border-line bg-surface px-4 py-3 text-[13px] text-bad">
            <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
            <div><span className="font-semibold">Could not run the scenario.</span> {error}</div>
          </div>
        )}
        {!data ? <Skeleton /> : (
          <div className={cx('space-y-4 transition-opacity', loading && 'opacity-60')} aria-busy={loading}>
            <Hero data={data} />
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <StrategyComparison data={data} />
              <ScheduleTable data={data} />
            </div>
            <HourlyCharts data={data} focus={shownFocus} onFocus={setFocus} />
            <DailyTable data={data} />
          </div>
        )}
      </main>
    </div>
  )
}
