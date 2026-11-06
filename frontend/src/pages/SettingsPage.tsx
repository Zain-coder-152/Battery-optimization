import { ArrowRight, RotateCcw } from 'lucide-react'
import type { Config, OptimizeRequest } from '../api'
import { GridSettings, HomeSettings } from '../components/Settings'

export default function SettingsPage({ config, value, onChange, firstVisit }: {
  config: Config
  value: OptimizeRequest
  onChange: (v: OptimizeRequest) => void
  firstVisit: boolean
}) {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{firstVisit ? 'Tell us about your home' : 'Settings'}</h1>
          <p className="mt-1 text-[14px] text-ink-2">
            {firstVisit ? 'Two quick steps. The defaults are a typical Lahore home, so you can also skip straight to the results.'
              : 'Changes apply right away; your settings are remembered on this device.'}
          </p>
        </div>
        <button type="button" onClick={() => onChange({ ...config.defaults, week: value.week })}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-muted hover:bg-surface-2 hover:text-ink">
          <RotateCcw className="size-3.5" aria-hidden />Reset to defaults
        </button>
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <HomeSettings config={config} value={value} onChange={onChange} />
        <GridSettings config={config} value={value} onChange={onChange} />
      </div>
      <div className="flex justify-end">
        <a href="#/" className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-[14px] font-semibold text-surface shadow-sm hover:opacity-90">
          See my results <ArrowRight className="size-4" aria-hidden />
        </a>
      </div>
    </div>
  )
}
