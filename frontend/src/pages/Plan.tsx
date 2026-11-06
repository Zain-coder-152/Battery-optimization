import type { OptimizeResponse, ScheduleRow } from '../api'
import { ScheduleTable } from '../components/Results'
import { num } from '../lib/format'

const SUMMARY: { action: ScheduleRow['action']; text: string }[] = [
  { action: 'Charge from solar', text: 'Store spare sunshine' },
  { action: 'Charge from grid', text: 'Buy cheap power for later' },
  { action: 'Discharge to home', text: 'Run the house on the battery' },
  { action: 'Hold', text: 'Leave the battery as it is' },
]

export default function Plan({ data }: { data: OptimizeResponse }) {
  const rows = data.schedule
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {SUMMARY.map(({ action, text }) => {
          const hrs = rows.filter((r) => r.action === action)
          const kwh = hrs.reduce((a, r) => a + Math.abs(r.battery_kwh), 0)
          return (
            <div key={action} className="rounded-xl border border-line bg-surface px-4 py-3.5">
              <div className="text-[13px] font-medium text-ink-2">{action}</div>
              <div className="mt-1.5 text-[22px] font-semibold leading-tight text-ink">{hrs.length} h</div>
              <div className="mt-0.5 text-[12px] text-muted">{action === 'Hold' ? text : `${text} · ${num(kwh)} kWh`}</div>
            </div>
          )
        })}
      </div>
      <ScheduleTable data={data} />
    </div>
  )
}
