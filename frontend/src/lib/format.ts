import type { StrategyId } from '../api'

const pkr0 = new Intl.NumberFormat('en-PK', { maximumFractionDigits: 0 })

export const pkr = (v: number) => `PKR ${pkr0.format(Math.round(v))}`
export const num = (v: number, digits = 1) =>
  new Intl.NumberFormat('en-PK', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v)
export const pct = (v: number, digits = 1, signed = false) =>
  `${signed && v > 0 ? '+' : ''}${num(v, digits)}%`
export const compact = (v: number) =>
  Math.abs(v) >= 1000 ? `${num(v / 1000, Math.abs(v) >= 10000 ? 0 : 1)}K` : pkr0.format(v)
export const hourLabel = (h: number) => `${String(h).padStart(2, '0')}:00`

/** The options shown to users, best first. The API also returns other strategies; they are not displayed. */
export const STRATEGY_ORDER: StrategyId[] = ['mpc', 'selfuse', 'none']

export const STRATEGY_LABEL: Record<StrategyId, string> = {
  mpc: 'Smart schedule',
  selfuse: 'Basic battery use',
  none: 'Without battery',
  learned: 'Learned',
  peakaware: 'Peak-only',
}

/** Identity colour per strategy: follows the entity, never its rank. */
export const STRATEGY_COLOR: Record<StrategyId, string> = {
  mpc: 'var(--s-mpc)',
  selfuse: 'var(--s-selfuse)',
  learned: 'var(--s-learned)',
  peakaware: 'var(--s-peakaware)',
  none: 'var(--s-none)',
}

/** Turn a boolean-per-hour array into contiguous [start, end] index ranges (for shaded bands). */
export function ranges(mask: boolean[]): [number, number][] {
  const out: [number, number][] = []
  let start = -1
  mask.forEach((m, i) => {
    if (m && start < 0) start = i
    if (!m && start >= 0) { out.push([start, i]); start = -1 }
  })
  if (start >= 0) out.push([start, mask.length])
  return out
}
