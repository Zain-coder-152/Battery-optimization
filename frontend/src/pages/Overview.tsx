import { ArrowRight } from 'lucide-react'
import type { OptimizeResponse } from '../api'
import { StrategyComparison } from '../components/Results'
import { Stat } from '../components/ui'
import { num, pct, pkr } from '../lib/format'

function Hero({ data }: { data: OptimizeResponse }) {
  const s = data.summary
  const mpc = s.mpc!, none = s.none!, self = s.selfuse!
  const saving = none.total_cost - mpc.total_cost
  const vsSelf = self.total_cost - mpc.total_cost
  const outages = data.request.outage_hours.length > 0
  return (
    <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr_1fr]">
      <div className="rounded-xl border border-line bg-surface px-5 py-5 lg:row-span-2">
        <div className="text-[13px] font-medium text-muted">A battery with the smart schedule saves you this week</div>
        <div className="mt-2 text-[44px] font-semibold leading-none tracking-tight text-ink">{pkr(saving)}</div>
        <div className="mt-2 text-[13px] text-ink-2">
          {none.total_cost > 0 && <>{pct(100 * saving / none.total_cost, 0)} lower bill · </>}≈ {pkr(saving * 52 / 12)} a month
        </div>
        <div className="mt-5 grid grid-cols-2 gap-3 border-t border-line pt-4 text-[13px]">
          <div>
            <div className="text-muted">Bill without battery</div>
            <div className="tnum mt-0.5 text-[16px] font-semibold text-ink">{pkr(none.total_cost)}</div>
          </div>
          <div>
            <div className="text-muted">Bill with smart schedule</div>
            <div className="tnum mt-0.5 text-[16px] font-semibold text-ink">{pkr(mpc.total_cost)}</div>
          </div>
        </div>
      </div>
      <Stat label="Compared with basic battery use" value={`${pkr(Math.abs(vsSelf))} ${vsSelf >= 0 ? 'less' : 'more'}`}
        sub={vsSelf >= 0.5 ? 'Smart schedule is cheaper' : vsSelf <= -0.5 ? 'Basic use is cheaper this week' : 'About the same'}
        tone={vsSelf >= 0.5 ? 'good' : vsSelf <= -0.5 ? 'bad' : undefined} />
      <Stat label="Self-sufficiency" value={pct(mpc.self_sufficiency_pct, 0)}
        sub={`of your use covered by your own solar and battery (no battery: ${pct(none.self_sufficiency_pct, 0)})`} />
      {outages ? (
        <Stat label="Power cut during outages" value={`${num(mpc.unserved_kwh)} kWh`}
          sub={`Basic battery use: ${num(self.unserved_kwh)} kWh · no battery: ${num(none.unserved_kwh)} kWh`}
          tone={mpc.unserved_kwh <= self.unserved_kwh ? 'good' : 'bad'} />
      ) : (
        <Stat label="Bought from the grid" value={`${num(mpc.import_kwh, 0)} kWh`} sub={`Without battery: ${num(none.import_kwh, 0)} kWh`} />
      )}
      <Stat label="Battery use" value={`${num(mpc.cycles)} full cycles`}
        sub={data.request.deg_cost > 0 ? `Wear ${pkr(mpc.wear_cost)}` : 'charge–discharge equivalents this week'} />
    </div>
  )
}

function Next({ href, title, text }: { href: string; title: string; text: string }) {
  return (
    <a href={href} className="group flex items-center gap-4 rounded-xl border border-line bg-surface px-5 py-4 transition hover:border-accent">
      <div className="min-w-0 flex-1">
        <div className="text-[14px] font-semibold text-ink">{title}</div>
        <div className="text-[13px] text-muted">{text}</div>
      </div>
      <ArrowRight className="size-4 shrink-0 text-muted transition group-hover:translate-x-0.5 group-hover:text-accent" aria-hidden />
    </a>
  )
}

export default function Overview({ data }: { data: OptimizeResponse }) {
  return (
    <div className="space-y-4">
      <Hero data={data} />
      <StrategyComparison data={data} />
      <div className="grid gap-3 md:grid-cols-2">
        <Next href="#/plan" title="Today's battery plan"
          text="Hour by hour: when to charge, when to use the battery." />
        <Next href="#/details" title="Details"
          text="Solar, usage and battery charts, plus the cost of each day." />
      </div>
    </div>
  )
}
