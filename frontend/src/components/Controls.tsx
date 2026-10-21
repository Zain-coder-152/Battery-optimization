import type { ReactNode } from 'react'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
import type { Config, OptimizeRequest } from '../api'
import { Field, HourGrid, NumberField, Segmented, Select, SliderField, cx } from './ui'
import { hourLabel } from '../lib/format'

function Group({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="space-y-3.5 border-b border-line px-5 py-5 last:border-b-0">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{title}</h3>
        {aside}
      </div>
      {children}
    </div>
  )
}

const sameHours = (a: number[], b: number[]) => a.length === b.length && a.every((h, i) => h === b[i])

function hoursSummary(h: number[]) {
  if (!h.length) return 'none'
  return `${h.length} h/day`
}

export default function Controls({ config, value, onChange }: {
  config: Config
  value: OptimizeRequest
  onChange: (v: OptimizeRequest) => void
}) {
  const set = <K extends keyof OptimizeRequest>(k: K, v: OptimizeRequest[K]) => onChange({ ...value, [k]: v })
  const d = config.defaults
  const t = value.tariff
  const isDefaultTariff = t.peak === config.default_tariff.peak && t.offpeak === config.default_tariff.offpeak &&
    t.export === config.default_tariff.export && sameHours(t.peak_hours, config.default_tariff.peak_hours)
  const presetName = Object.entries(config.outage_presets).find(([, hrs]) => sameHours(hrs, value.outage_hours))?.[0] ?? 'custom'
  const loc = config.locations.find((l) => l.id === value.location)

  return (
    <div className="text-ink">
      <Group title="Location & period" aside={
        <button type="button" onClick={() => onChange(d)}
          className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[12px] font-medium text-muted hover:bg-surface-2 hover:text-ink">
          <RotateCcw className="size-3.5" aria-hidden /> Reset all
        </button>
      }>
        <Field label="City" htmlFor="loc" hint={loc && `${loc.annual_kwh_per_5kw.toLocaleString()} kWh/yr per 5 kW`}>
          <Select id="loc" value={value.location} onChange={(v) => set('location', v)}>
            {config.locations.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
          </Select>
        </Field>
        <Field label="Week" htmlFor="week" hint={`Week ${value.week + 1} of 52`}>
          <div className="flex gap-1.5">
            <button type="button" aria-label="Previous week" disabled={value.week === 0}
              onClick={() => set('week', value.week - 1)}
              className="grid w-9 shrink-0 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2 disabled:opacity-40">
              <ChevronLeft className="size-4" />
            </button>
            <Select id="week" value={value.week} onChange={(v) => set('week', Number(v))}>
              {config.weeks.map((w) => <option key={w.index} value={w.index}>{w.label}</option>)}
            </Select>
            <button type="button" aria-label="Next week" disabled={value.week === config.weeks.length - 1}
              onClick={() => set('week', value.week + 1)}
              className="grid w-9 shrink-0 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2 disabled:opacity-40">
              <ChevronRight className="size-4" />
            </button>
          </div>
        </Field>
      </Group>

      <Group title="System">
        <SliderField id="panel" label="Solar panels" unit="kW" value={value.panel_kw} min={1} max={20} step={0.5}
          onChange={(v) => set('panel_kw', v)} />
        <SliderField id="battery" label="Battery capacity" unit="kWh" value={value.battery_kwh} min={2} max={40} step={1}
          onChange={(v) => set('battery_kwh', v)} />
        <SliderField id="power" label="Inverter power" unit="kW" value={value.max_power_kw} min={1} max={10} step={0.5}
          onChange={(v) => set('max_power_kw', v)} />
        <SliderField id="demand" label="Household demand" unit="×" value={value.demand_scale} min={0.5} max={4} step={0.1}
          onChange={(v) => set('demand_scale', v)} />
      </Group>

      <Group title="Tariff (PKR/kWh)" aside={isDefaultTariff
        ? <span className="text-[12px] text-muted">NEPRA time-of-use</span>
        : (
          <button type="button" onClick={() => set('tariff', config.default_tariff)}
            className="rounded-md px-1.5 py-0.5 text-[12px] font-medium text-accent hover:bg-accent-soft">
            Restore NEPRA rates
          </button>
        )}>
        <div className="grid grid-cols-3 gap-2">
          <NumberField id="peak" label="Peak" unit="" value={t.peak} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, peak: v })} />
          <NumberField id="offpeak" label="Off-peak" unit="" value={t.offpeak} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, offpeak: v })} />
          <NumberField id="export" label="Export" unit="" value={t.export} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, export: v })} />
        </div>
        <Field label="Peak hours" hint={t.peak_hours.length ? `${hourLabel(t.peak_hours[0])} start · ${t.peak_hours.length} h` : 'flat rate'}>
          <HourGrid label="Peak price hours" tone="peak" value={t.peak_hours}
            onChange={(h) => set('tariff', { ...t, peak_hours: h })} />
        </Field>
      </Group>

      <Group title="Load-shedding">
        <Segmented label="Load-shedding preset" value={presetName as 'none' | 'light' | 'heavy' | 'custom'}
          options={[{ value: 'none', label: 'None' }, { value: 'light', label: '2 h' }, { value: 'heavy', label: '6 h' },
            { value: 'custom', label: 'Custom' }]}
          onChange={(v) => { if (v !== 'custom') set('outage_hours', config.outage_presets[v]) }} />
        <Field label="Grid off at" hint={hoursSummary(value.outage_hours)}>
          <HourGrid label="Load-shedding hours" tone="outage" value={value.outage_hours}
            onChange={(h) => set('outage_hours', h)} />
        </Field>
        <p className="text-[12px] leading-relaxed text-muted">
          Hours when the grid is off every day. The smart schedule keeps charge in reserve for them.
        </p>
      </Group>

      <Group title="Battery wear">
        <SliderField id="wear" label="Wear cost" unit="PKR/kWh" value={value.deg_cost} min={0} max={15} step={0.5}
          onChange={(v) => set('deg_cost', v)} />
        <p className={cx('text-[12px] leading-relaxed text-muted')}>
          Cost of each kWh moved through the battery (pack price ÷ lifetime throughput). ~3 PKR/kWh for LFP.
        </p>
      </Group>
    </div>
  )
}
