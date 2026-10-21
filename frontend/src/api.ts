export type StrategyId = 'none' | 'selfuse' | 'peakaware' | 'learned' | 'mpc'

export interface Location {
  id: string
  name: string
  lat: number
  lon: number
  annual_kwh_per_5kw: number
}

export interface Tariff {
  peak: number
  offpeak: number
  export: number
  peak_hours: number[]
}

export interface OptimizeRequest {
  location: string
  panel_kw: number
  battery_kwh: number
  max_power_kw: number
  week: number
  demand_scale: number
  tariff: Tariff
  outage_hours: number[]
  deg_cost: number
}

export interface Config {
  locations: Location[]
  default_tariff: Tariff
  strategies: Record<StrategyId, string>
  weeks: { index: number; start: string; label: string }[]
  defaults: OptimizeRequest
  outage_presets: Record<string, number[]>
}

export interface StrategySummary {
  id: StrategyId
  name: string
  grid_cost: number
  wear_cost: number
  total_cost: number
  import_kwh: number
  export_kwh: number
  unserved_kwh: number
  self_sufficiency_pct: number
  cycles: number
  saving_vs_none_pct: number | null
}

export interface Series {
  soc: number[]
  flow: number[]
  import: number[]
  export: number[]
  unserved: number[]
}

export interface ScheduleRow {
  time: string
  hour: number
  action: 'Charge from grid' | 'Charge from solar' | 'Discharge to home' | 'Hold'
  battery_kwh: number
  soc_kwh: number
  grid_kwh: number
  price: number
  outage: boolean
}

export interface OptimizeResponse {
  meta: {
    location: string
    period: string
    hours: number
    learned_available: boolean
    learned_reason: string
    recommended: StrategyId
    runtime_ms: number
  }
  request: OptimizeRequest
  summary: Partial<Record<StrategyId, StrategySummary>>
  hourly: {
    time: string[]
    solar: number[]
    demand: number[]
    price: number[]
    outage: boolean[]
    series: Partial<Record<StrategyId, Series>>
  }
  schedule: ScheduleRow[]
  daily: ({ date: string } & Partial<Record<StrategyId, number>>)[]
}

async function request<T>(path: string, init?: RequestInit, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { ...init, signal })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* keep the status text */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  config: () => request<Config>('/api/config'),
  optimize: (body: OptimizeRequest, signal?: AbortSignal) =>
    request<OptimizeResponse>('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }, signal),
}
