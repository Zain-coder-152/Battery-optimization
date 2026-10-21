import { useMemo, useState } from 'react'
import {
  Area, Bar, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { LineChart as LineIcon } from 'lucide-react'
import type { OptimizeResponse, StrategyId } from '../api'
import { Card, Legend, Segmented, Swatch } from './ui'
import { STRATEGY_COLOR, STRATEGY_LABEL, num, ranges } from '../lib/format'

const SYNC = 'hourly'
const axisTick = { fill: 'var(--muted)', fontSize: 11 }

type Row = Record<string, number | string | boolean>

function dayTicks(times: string[]) {
  return times.map((_, i) => i).filter((i) => i % 24 === 0)
}

function Bands({ peak, outage }: { peak: [number, number][]; outage: [number, number][] }) {
  return (
    <>
      {peak.map(([a, b]) => <ReferenceArea key={`p${a}`} x1={a - 0.5} x2={b - 0.5} fill="var(--band-peak)" strokeOpacity={0} ifOverflow="hidden" />)}
      {outage.map(([a, b]) => <ReferenceArea key={`o${a}`} x1={a - 0.5} x2={b - 0.5} fill="var(--band-outage)" strokeOpacity={0} ifOverflow="hidden" />)}
    </>
  )
}

interface TipItem { key: string; label: string; color: string; unit: string; line?: boolean }

function Tip({ active, payload, rows, items }: {
  active?: boolean
  payload?: { payload: Row }[]
  rows: Row[]
  items: TipItem[]
}) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  const row = rows[r.i as number]
  const date = new Date(String(row.time))
  return (
    <div className="min-w-[180px] rounded-lg border border-line bg-surface px-3 py-2.5 text-[12px] shadow-lg">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="font-semibold text-ink">
          {date.toLocaleDateString('en', { weekday: 'short', day: 'numeric', month: 'short' })} · {date.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: false })}
        </span>
      </div>
      <div className="space-y-1">
        {items.map((it) => (
          <div key={it.key} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-2"><Swatch color={it.color} line={it.line} />{it.label}</span>
            <span className="tnum font-medium text-ink">{num(Number(row[it.key]), 2)} {it.unit}</span>
          </div>
        ))}
        <div className="mt-1.5 flex items-center justify-between gap-4 border-t border-line pt-1.5 text-ink-2">
          <span>Grid price</span>
          <span className="tnum">{row.outage ? 'Grid off' : `${num(Number(row.price), 2)} PKR`}</span>
        </div>
      </div>
    </div>
  )
}

function Frame({ height, children }: { height: number; children: React.ReactElement }) {
  return <div style={{ height }} className="w-full">
    <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
  </div>
}

export default function HourlyCharts({ data, focus, onFocus }: {
  data: OptimizeResponse
  focus: StrategyId
  onFocus: (s: StrategyId) => void
}) {
  const [view, setView] = useState<'chart' | 'table'>('chart')
  const h = data.hourly
  const shown = (['mpc', 'selfuse'] as StrategyId[]).filter((s) => h.series[s])
  const peakMask = useMemo(() => {
    const max = Math.max(...h.price), min = Math.min(...h.price)
    return h.price.map((p) => max > min && p >= max - 1e-9)
  }, [h.price])
  const peak = useMemo(() => ranges(peakMask), [peakMask])
  const outage = useMemo(() => ranges(h.outage), [h.outage])
  const ticks = useMemo(() => dayTicks(h.time), [h.time])

  const rows: Row[] = useMemo(() => h.time.map((time, i) => {
    const r: Row = { i, time, solar: h.solar[i], demand: h.demand[i], price: h.price[i], outage: h.outage[i] }
    for (const s of Object.keys(h.series) as StrategyId[]) {
      const ser = h.series[s]!
      r[`soc_${s}`] = ser.soc[i]
      r[`flow_${s}`] = ser.flow[i]
      r[`charge_${s}`] = Math.max(ser.flow[i], 0)
      r[`discharge_${s}`] = Math.min(ser.flow[i], 0)
      r[`grid_${s}`] = ser.import[i] - ser.export[i]
    }
    return r
  }), [h])

  const tickFmt = (i: number) => new Date(String(rows[i]?.time ?? h.time[0])).toLocaleDateString('en', { weekday: 'short', day: 'numeric' })
  const xAxis = <XAxis dataKey="i" type="number" domain={[-0.5, rows.length - 0.5]} ticks={ticks} tickFormatter={tickFmt}
    tick={axisTick} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} />
  const grid = <CartesianGrid stroke="var(--grid)" vertical={false} />
  const cursor = { stroke: 'var(--axis)', strokeWidth: 1 }
  const bandLegend = [
    ...(peak.length ? [{ label: 'Peak price hours', color: 'var(--band-peak)' }] : []),
    ...(outage.length ? [{ label: 'Grid off (load-shedding)', color: 'var(--band-outage)' }] : []),
  ]
  const name = (s: StrategyId) => STRATEGY_LABEL[s]
  const cap = data.request.battery_kwh

  return (
    <Card title="Hour by hour" subtitle={`${data.meta.location} · ${data.meta.period}`}
      actions={
        <div className="w-[180px]">
          <Segmented label="View" value={view} onChange={setView}
            options={[{ value: 'chart', label: 'Charts' }, { value: 'table', label: 'Table' }]} />
        </div>
      }>
      {view === 'chart' ? (
        <div className="space-y-6">
          <div>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 className="flex items-center gap-1.5 text-[13px] font-semibold text-ink"><LineIcon className="size-3.5 text-muted" aria-hidden />Solar output and household demand (kWh)</h3>
              <Legend items={[{ label: 'Solar', color: 'var(--s-solar)', line: true }, { label: 'Demand', color: 'var(--s-demand)', line: true }, ...bandLegend]} />
            </div>
            <Frame height={200}>
              <ComposedChart data={rows} syncId={SYNC} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                {grid}<Bands peak={peak} outage={outage} />{xAxis}
                <YAxis tick={axisTick} tickLine={false} axisLine={false} width={44} />
                <Tooltip cursor={cursor} content={<Tip rows={rows} items={[
                  { key: 'solar', label: 'Solar', color: 'var(--s-solar)', unit: 'kWh', line: true },
                  { key: 'demand', label: 'Demand', color: 'var(--s-demand)', unit: 'kWh', line: true }]} />} />
                <Area type="monotone" dataKey="solar" stroke="var(--s-solar)" strokeWidth={2} fill="var(--s-solar)" fillOpacity={0.1} isAnimationActive={false} activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} />
                <Line type="monotone" dataKey="demand" stroke="var(--s-demand)" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} />
              </ComposedChart>
            </Frame>
          </div>

          <div>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-[13px] font-semibold text-ink">Battery charge level (kWh of {num(cap, 0)})</h3>
              <Legend items={shown.map((s) => ({ label: name(s), color: STRATEGY_COLOR[s], line: true }))} />
            </div>
            <Frame height={200}>
              <ComposedChart data={rows} syncId={SYNC} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                {grid}<Bands peak={peak} outage={outage} />{xAxis}
                <YAxis tick={axisTick} tickLine={false} axisLine={false} width={44} domain={[0, Math.ceil(cap)]} />
                <Tooltip cursor={cursor} content={<Tip rows={rows} items={shown.map((s) => ({ key: `soc_${s}`, label: name(s), color: STRATEGY_COLOR[s], unit: 'kWh', line: true }))} />} />
                {shown.slice().reverse().map((s) => (
                  <Line key={s} type="linear" dataKey={`soc_${s}`} stroke={STRATEGY_COLOR[s]} strokeWidth={s === 'mpc' ? 2.25 : 1.75}
                    dot={false} isAnimationActive={false} activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} />
                ))}
              </ComposedChart>
            </Frame>
          </div>

          <div>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-[13px] font-semibold text-ink">Battery power: charging above zero, discharging below (kWh)</h3>
              <div className="w-full sm:w-auto">
                <Segmented label="Strategy" value={focus} onChange={onFocus}
                  options={shown.map((s) => ({ value: s, label: STRATEGY_LABEL[s] }))} />
              </div>
            </div>
            <Frame height={170}>
              <ComposedChart data={rows} syncId={SYNC} margin={{ top: 4, right: 8, bottom: 0, left: -12 }} barCategoryGap={1}>
                {grid}<Bands peak={peak} outage={outage} />{xAxis}
                <YAxis tick={axisTick} tickLine={false} axisLine={false} width={44} />
                <ReferenceLine y={0} stroke="var(--axis)" />
                <Tooltip cursor={{ fill: 'var(--band-outage)' }} content={<Tip rows={rows} items={[
                  { key: `flow_${focus}`, label: `${name(focus)} battery`, color: STRATEGY_COLOR[focus], unit: 'kWh' },
                  { key: `grid_${focus}`, label: 'Net grid import', color: 'var(--s-none)', unit: 'kWh' }]} />} />
                <Bar dataKey={`charge_${focus}`} stackId="f" fill={STRATEGY_COLOR[focus]} isAnimationActive={false} radius={[2, 2, 0, 0]} />
                <Bar dataKey={`discharge_${focus}`} stackId="f" fill={STRATEGY_COLOR[focus]} fillOpacity={0.55} isAnimationActive={false} radius={[0, 0, 2, 2]} />
              </ComposedChart>
            </Frame>
          </div>
        </div>
      ) : (
        <div className="max-h-[560px] overflow-auto rounded-lg border border-line">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 bg-surface-2 text-left text-muted">
              <tr>
                {['Time', 'Solar', 'Demand', 'Price', ...shown.map((s) => `${name(s)} level`), `${name(focus)} power`].map((c) => (
                  <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody className="tnum">
              {rows.map((r) => (
                <tr key={String(r.time)} className="border-t border-line">
                  <td className="whitespace-nowrap px-3 py-1.5 text-ink-2">{String(r.time).replace('T', ' ')}</td>
                  <td className="px-3 py-1.5">{num(Number(r.solar), 2)}</td>
                  <td className="px-3 py-1.5">{num(Number(r.demand), 2)}</td>
                  <td className="px-3 py-1.5">{r.outage ? 'Grid off' : num(Number(r.price), 2)}</td>
                  {shown.map((s) => <td key={s} className="px-3 py-1.5">{num(Number(r[`soc_${s}`]), 2)}</td>)}
                  <td className="px-3 py-1.5">{num(Number(r[`flow_${focus}`]), 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

