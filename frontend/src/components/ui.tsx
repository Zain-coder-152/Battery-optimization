import type { ReactNode } from 'react'
import { Info } from 'lucide-react'

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(' ')
}

export function Card({ title, subtitle, actions, children, className }: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cx('rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.04)]', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-4">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5 pt-4">{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone }: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: 'good' | 'bad'
}) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="text-[12px] font-medium text-muted">{label}</div>
      <div className="mt-1 text-[22px] font-semibold leading-tight text-ink">{value}</div>
      {sub && (
        <div className={cx('mt-1 text-[12px]', tone === 'good' ? 'text-good' : tone === 'bad' ? 'text-bad' : 'text-ink-2')}>
          {sub}
        </div>
      )}
    </div>
  )
}

export function Field({ label, hint, children, htmlFor }: {
  label: string
  hint?: ReactNode
  children: ReactNode
  htmlFor?: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="text-[13px] font-medium text-ink-2">{label}</label>
        {hint && <span className="text-[12px] text-muted">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

const inputCls =
  'w-full rounded-lg border border-line bg-surface px-3 py-2 text-[14px] text-ink outline-none transition ' +
  'focus:border-accent focus:ring-2 focus:ring-accent/25'

export function Select({ id, value, onChange, children }: {
  id?: string
  value: string | number
  onChange: (v: string) => void
  children: ReactNode
}) {
  return (
    <select id={id} className={cx(inputCls, 'cursor-pointer appearance-none bg-[length:16px] bg-[right_10px_center] bg-no-repeat pr-8')}
      style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23898781' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")" }}
      value={value} onChange={(e) => onChange(e.target.value)}>
      {children}
    </select>
  )
}

/** Slider with a linked number box. */
export function SliderField({ id, label, unit, value, min, max, step, onChange }: {
  id: string
  label: string
  unit: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <Field label={label} htmlFor={id}>
      <div className="flex items-center gap-3">
        <input type="range" className="h-1.5 min-w-0 flex-1 cursor-pointer" min={min} max={max} step={step}
          value={value} aria-label={label} onChange={(e) => onChange(Number(e.target.value))} />
        <div className="flex w-[104px] shrink-0 items-center rounded-lg border border-line bg-surface focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
          <input id={id} type="number" className="tnum w-full min-w-0 bg-transparent py-1.5 pl-2.5 text-right text-[14px] text-ink outline-none"
            min={min} max={max} step={step} value={value}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)))
            }} />
          <span className="px-2 text-[12px] text-muted">{unit}</span>
        </div>
      </div>
    </Field>
  )
}

export function NumberField({ id, label, unit, value, min, max, step, onChange }: {
  id: string
  label: string
  unit: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <Field label={label} htmlFor={id}>
      <div className="flex items-center rounded-lg border border-line bg-surface focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
        <input id={id} type="number" className="tnum w-full min-w-0 bg-transparent px-2.5 py-2 text-[14px] text-ink outline-none"
          min={min} max={max} step={step} value={value}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)))
          }} />
        {unit && <span className="whitespace-nowrap pr-2.5 text-[12px] text-muted">{unit}</span>}
      </div>
    </Field>
  )
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded-lg bg-surface-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx('flex-1 rounded-md px-2.5 py-1.5 text-[13px] font-medium transition',
            value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** 24 toggles, one per hour of the day. */
export function HourGrid({ value, onChange, label, tone }: {
  value: number[]
  onChange: (v: number[]) => void
  label: string
  tone: 'peak' | 'outage'
}) {
  const on = new Set(value)
  const activeCls = tone === 'peak'
    ? 'bg-[var(--s-selfuse)] text-white border-transparent'
    : 'bg-ink text-surface border-transparent'
  return (
    <div role="group" aria-label={label}>
      <div className="grid grid-cols-12 gap-1">
        {Array.from({ length: 24 }, (_, h) => (
          <button key={h} type="button" aria-pressed={on.has(h)} title={`${String(h).padStart(2, '0')}:00–${String((h + 1) % 24).padStart(2, '0')}:00`}
            onClick={() => {
              const next = new Set(on)
              if (next.has(h)) next.delete(h); else next.add(h)
              onChange([...next].sort((a, b) => a - b))
            }}
            className={cx('tnum h-7 rounded-md border text-[11px] font-medium transition',
              on.has(h) ? activeCls : 'border-line bg-surface text-muted hover:border-accent hover:text-ink')}>
            {h}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'warn' | 'good' }) {
  const cls = {
    neutral: 'bg-surface-2 text-ink-2',
    accent: 'bg-accent-soft text-accent',
    warn: 'bg-warn-bg text-warn-ink',
    good: 'bg-surface-2 text-good',
  }[tone]
  return <span className={cx('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold', cls)}>{children}</span>
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-2.5 rounded-lg bg-surface-2 px-3.5 py-2.5 text-[13px] leading-relaxed text-ink-2">
      <Info className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
      <div>{children}</div>
    </div>
  )
}

export function Swatch({ color, line }: { color: string; line?: boolean }) {
  return line
    ? <span className="inline-block h-[3px] w-3.5 rounded-full" style={{ background: color }} aria-hidden />
    : <span className="inline-block size-2.5 rounded-[3px]" style={{ background: color }} aria-hidden />
}

export function Legend({ items }: { items: { label: string; color: string; line?: boolean }[] }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[12px] text-ink-2">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5"><Swatch color={i.color} line={i.line} />{i.label}</li>
      ))}
    </ul>
  )
}
