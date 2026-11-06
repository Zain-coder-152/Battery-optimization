import { useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import type { Config, OptimizeRequest } from '../api'
import { Field, HourGrid, NumberField, Segmented, Select, SliderField, cx } from './ui'
import { hourLabel } from '../lib/format'

type Props = { config: Config; value: OptimizeRequest; onChange: (v: OptimizeRequest) => void }

const sameHours = (a: number[], b: number[]) => a.length === b.length && a.every((h, i) => h === b[i])

function Section({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface">
      <header className="border-b border-line px-5 py-4">
        <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
        <p className="text-[13px] text-muted">{subtitle}</p>
      </header>
      <div className="space-y-5 px-5 py-5">{children}</div>
    </section>
  )
}

function Sub({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-3.5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{title}</h3>
        {aside}
      </div>
      {children}
    </div>
  )
}

export function HomeSettings({ config, value, onChange }: Props) {
  const set = <K extends keyof OptimizeRequest>(k: K, v: OptimizeRequest[K]) => onChange({ ...value, [k]: v })
  const loc = config.locations.find((l) => l.id === value.location)
  return (
    <Section title="Your home" subtitle="Where you live, how much you use and the system you have.">
      <Field label="City" htmlFor="loc" hint={loc && `${loc.annual_kwh_per_5kw.toLocaleString()} kWh/yr per 5 kW of panels`}>
        <Select id="loc" value={value.location} onChange={(v) => set('location', v)}>
          {config.locations.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
        </Select>
      </Field>
      <div>
        <SliderField id="demand" label="Daily electricity use" unit="kWh" value={value.daily_kwh} min={2} max={80} step={1}
          onChange={(v) => set('daily_kwh', v)} />
        {loc && (
          <p className="mt-1.5 text-[12px] text-muted">
            A typical {loc.name} home uses {loc.typical_kwh_per_day} kWh/day. Your bill divided by the days in it gives yours.{' '}
            {value.daily_kwh !== loc.typical_kwh_per_day && (
              <button type="button" onClick={() => set('daily_kwh', loc.typical_kwh_per_day)}
                className="font-medium text-accent hover:underline">Use typical</button>
            )}
          </p>
        )}
      </div>
      <SliderField id="panel" label="Solar panels" unit="kW" value={value.panel_kw} min={1} max={20} step={0.5}
        onChange={(v) => set('panel_kw', v)} />
      <SliderField id="battery" label="Battery size" unit="kWh" value={value.battery_kwh} min={2} max={40} step={1}
        onChange={(v) => set('battery_kwh', v)} />
      <SliderField id="power" label="Inverter power" unit="kW" value={value.max_power_kw} min={1} max={10} step={0.5}
        onChange={(v) => set('max_power_kw', v)} />
    </Section>
  )
}

export function GridSettings({ config, value, onChange }: Props) {
  const [advanced, setAdvanced] = useState(value.deg_cost > 0)
  const set = <K extends keyof OptimizeRequest>(k: K, v: OptimizeRequest[K]) => onChange({ ...value, [k]: v })
  const t = value.tariff
  const isDefaultTariff = t.peak === config.default_tariff.peak && t.offpeak === config.default_tariff.offpeak &&
    t.export === config.default_tariff.export && sameHours(t.peak_hours, config.default_tariff.peak_hours)
  const preset = Object.entries(config.outage_presets).find(([, hrs]) => sameHours(hrs, value.outage_hours))?.[0] ?? 'custom'

  return (
    <Section title="Electricity and outages" subtitle="Your tariff and any daily load-shedding.">
      <Sub title="Tariff, PKR per kWh" aside={isDefaultTariff
        ? <span className="text-[12px] text-muted">NEPRA time-of-use</span>
        : (
          <button type="button" onClick={() => set('tariff', config.default_tariff)}
            className="rounded-md px-1.5 py-0.5 text-[12px] font-medium text-accent hover:bg-accent-soft">Restore NEPRA rates</button>
        )}>
        <div className="grid grid-cols-3 gap-2">
          <NumberField id="peak" label="Peak" unit="" value={t.peak} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, peak: v })} />
          <NumberField id="offpeak" label="Off-peak" unit="" value={t.offpeak} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, offpeak: v })} />
          <NumberField id="export" label="Export" unit="" value={t.export} min={0} max={200} step={0.5}
            onChange={(v) => set('tariff', { ...t, export: v })} />
        </div>
        <Field label="Peak hours" hint={t.peak_hours.length ? `from ${hourLabel(t.peak_hours[0])} · ${t.peak_hours.length} h` : 'flat rate'}>
          <HourGrid label="Peak price hours" tone="peak" value={t.peak_hours} onChange={(h) => set('tariff', { ...t, peak_hours: h })} />
        </Field>
      </Sub>

      <div className="border-t border-line" />

      <Sub title="Load-shedding">
        <Segmented label="Load-shedding" value={preset as 'none' | 'light' | 'heavy' | 'custom'}
          options={[{ value: 'none', label: 'None' }, { value: 'light', label: '2 h a day' }, { value: 'heavy', label: '6 h a day' },
            { value: 'custom', label: 'Custom' }]}
          onChange={(v) => { if (v !== 'custom') set('outage_hours', config.outage_presets[v]) }} />
        <Field label="Grid off at" hint={value.outage_hours.length ? `${value.outage_hours.length} h a day` : 'none'}>
          <HourGrid label="Load-shedding hours" tone="outage" value={value.outage_hours} onChange={(h) => set('outage_hours', h)} />
        </Field>
        <p className="text-[12px] leading-relaxed text-muted">Tap the hours your area loses power. The smart schedule keeps charge in reserve for them.</p>
      </Sub>

      <div className="border-t border-line" />

      <div>
        <button type="button" onClick={() => setAdvanced(!advanced)} aria-expanded={advanced}
          className="flex w-full items-center justify-between text-[11px] font-semibold uppercase tracking-[0.08em] text-muted hover:text-ink">
          Advanced
          <ChevronDown className={cx('size-4 transition-transform', advanced && 'rotate-180')} aria-hidden />
        </button>
        {advanced && (
          <div className="mt-3.5 space-y-2">
            <SliderField id="wear" label="Battery wear cost" unit="PKR/kWh" value={value.deg_cost} min={0} max={15} step={0.5}
              onChange={(v) => set('deg_cost', v)} />
            <p className="text-[12px] leading-relaxed text-muted">
              What each kWh through the battery costs in lifetime (pack price ÷ lifetime throughput). About 3 PKR/kWh for LFP. Leave at 0 to ignore.
            </p>
          </div>
        )}
      </div>
    </Section>
  )
}
