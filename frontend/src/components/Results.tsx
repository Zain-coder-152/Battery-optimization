import { ArrowDownToLine, ArrowUpFromLine, BatteryCharging, Pause, PlugZap, Sun, TriangleAlert } from 'lucide-react'
import type { OptimizeResponse, ScheduleRow, StrategyId, StrategySummary } from '../api'
import { Badge, Card, Swatch, cx } from './ui'
import { STRATEGY_COLOR, STRATEGY_LABEL, STRATEGY_ORDER, hourLabel, num, pct, pkr } from '../lib/format'

/** Horizontal bars, one per strategy, sorted cheapest first. */
export function StrategyComparison({ data }: { data: OptimizeResponse }) {
  const rows = STRATEGY_ORDER.map((s) => data.summary[s]).filter(Boolean) as StrategySummary[]
  rows.sort((a, b) => a.total_cost - b.total_cost)
  const max = Math.max(...rows.map((r) => Math.max(r.total_cost, 0)), 1)
  const cheapest = rows[0]?.id
  const outages = data.request.outage_hours.length > 0
  const wear = data.request.deg_cost > 0

  return (
    <Card title="Weekly cost comparison" subtitle={`Grid bill${wear ? ' + battery wear' : ''}, PKR for the selected week. Lower is better.`}>
      <ul className="space-y-3.5">
        {rows.map((r) => (
          <li key={r.id}>
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
              <span className="flex items-center gap-2 text-[13px] font-medium text-ink">
                <Swatch color={STRATEGY_COLOR[r.id]} />{STRATEGY_LABEL[r.id]}
                {r.id === cheapest && <Badge tone="good">Cheapest this week</Badge>}
              </span>
              <span className="tnum text-[13px] text-ink">
                <span className="font-semibold">{pkr(r.total_cost)}</span>
                {r.id !== 'none' && r.saving_vs_none_pct !== null && (
                  <span className="ml-2 text-muted">saves {pct(r.saving_vs_none_pct)}</span>
                )}
              </span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-surface-2">
              <div className="h-full rounded-full transition-[width] duration-500"
                style={{ width: `${Math.max(0, (r.total_cost / max) * 100)}%`, background: STRATEGY_COLOR[r.id] }} />
            </div>
            <div className="tnum mt-1 flex flex-wrap gap-x-4 text-[12px] text-muted">
              <span>Import {num(r.import_kwh)} kWh</span>
              <span>Export {num(r.export_kwh)} kWh</span>
              <span>Self-sufficiency {pct(r.self_sufficiency_pct, 0)}</span>
              {r.id !== 'none' && <span>{num(r.cycles)} cycles</span>}
              {wear && r.wear_cost > 0 && <span>Wear {pkr(r.wear_cost)}</span>}
              {outages && (
                <span className={cx(r.unserved_kwh > 0.05 ? 'font-medium text-bad' : '')}>
                  Unmet during outages {num(r.unserved_kwh)} kWh
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </Card>
  )
}

const ACTION: Record<ScheduleRow['action'], { icon: typeof Sun; tone: string }> = {
  'Charge from solar': { icon: Sun, tone: 'text-[var(--s-peakaware)]' },
  'Charge from grid': { icon: PlugZap, tone: 'text-accent' },
  'Discharge to home': { icon: BatteryCharging, tone: 'text-[var(--s-selfuse)]' },
  Hold: { icon: Pause, tone: 'text-muted' },
}

export function ScheduleTable({ data }: { data: OptimizeResponse }) {
  const rows = data.schedule
  const first = rows[0] && new Date(rows[0].time)
  return (
    <Card title="Recommended schedule"
      subtitle={`Battery actions for the first 24 hours${first ? ` (${first.toLocaleDateString('en', { weekday: 'long', day: 'numeric', month: 'short' })})` : ''}.`}>
      <div className="max-h-[440px] overflow-auto rounded-lg border border-line">
        <table className="w-full text-[13px]">
          <thead className="sticky top-0 z-10 bg-surface-2 text-left text-[12px] text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">Hour</th>
              <th className="px-3 py-2 font-medium">Action</th>
              <th className="px-3 py-2 text-right font-medium">Battery</th>
              <th className="px-3 py-2 text-right font-medium">Level</th>
              <th className="px-3 py-2 text-right font-medium">Grid</th>
              <th className="px-3 py-2 text-right font-medium">Price</th>
            </tr>
          </thead>
          <tbody className="tnum">
            {rows.map((r) => {
              const A = ACTION[r.action]
              return (
                <tr key={r.time} className="border-t border-line">
                  <td className="whitespace-nowrap px-3 py-2 text-ink-2">{hourLabel(r.hour)}</td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className="flex items-center gap-1.5 text-ink">
                      <A.icon className={cx('size-3.5', A.tone)} aria-hidden />{r.action}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-ink">{r.battery_kwh > 0 ? '+' : ''}{num(r.battery_kwh, 2)}</td>
                  <td className="px-3 py-2 text-right text-ink-2">{num(r.soc_kwh, 1)}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-right text-ink-2">
                    {r.outage ? <Badge tone="warn"><TriangleAlert className="size-3" aria-hidden />Off</Badge> : (
                      <span className="inline-flex items-center gap-1">
                        {r.grid_kwh > 0.005 ? <ArrowDownToLine className="size-3 text-muted" aria-label="import" />
                          : r.grid_kwh < -0.005 ? <ArrowUpFromLine className="size-3 text-muted" aria-label="export" /> : null}
                        {num(Math.abs(r.grid_kwh), 2)}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right text-ink-2">{r.outage ? '—' : num(r.price, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[12px] text-muted">Battery and grid in kWh for that hour; level in kWh after the hour. Price in PKR/kWh.</p>
    </Card>
  )
}

export function DailyTable({ data }: { data: OptimizeResponse }) {
  const cols = STRATEGY_ORDER.filter((s) => data.summary[s]) as StrategyId[]
  const totals = Object.fromEntries(cols.map((s) => [s, data.daily.reduce((a, d) => a + (d[s] ?? 0), 0)]))
  return (
    <Card title="Cost per day" subtitle="PKR, grid bill plus any battery wear. Negative means the home earned more from exports than it paid.">
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full text-[13px]">
          <thead className="bg-surface-2 text-left text-[12px] text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">Day</th>
              {cols.map((s) => (
                <th key={s} className="whitespace-nowrap px-3 py-2 text-right font-medium">
                  <span className="inline-flex items-center gap-1.5"><Swatch color={STRATEGY_COLOR[s]} />{STRATEGY_LABEL[s]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="tnum">
            {data.daily.map((d) => {
              const best = Math.min(...cols.map((s) => d[s] ?? Infinity))
              return (
                <tr key={d.date} className="border-t border-line">
                  <td className="whitespace-nowrap px-3 py-2 text-ink-2">{d.date}</td>
                  {cols.map((s) => (
                    <td key={s} className={cx('px-3 py-2 text-right', d[s] === best ? 'font-semibold text-ink' : 'text-ink-2')}>
                      {num(d[s] ?? 0, 0)}
                    </td>
                  ))}
                </tr>
              )
            })}
            <tr className="border-t-2 border-line bg-surface-2/60 font-semibold">
              <td className="px-3 py-2 text-ink">Week</td>
              {cols.map((s) => <td key={s} className="px-3 py-2 text-right text-ink">{num(totals[s], 0)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[12px] text-muted">Bold marks the cheapest option each day.</p>
    </Card>
  )
}
