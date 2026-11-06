import { useState } from 'react'
import type { OptimizeResponse, StrategyId } from '../api'
import HourlyCharts from '../components/HourlyCharts'
import { DailyTable } from '../components/Results'

export default function Details({ data }: { data: OptimizeResponse }) {
  const [focus, setFocus] = useState<StrategyId>('mpc')
  const shown: StrategyId = data.hourly.series[focus] ? focus : 'mpc'
  return (
    <div className="space-y-4">
      <HourlyCharts data={data} focus={shown} onFocus={setFocus} />
      <DailyTable data={data} />
    </div>
  )
}
