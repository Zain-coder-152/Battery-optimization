import { useEffect, useState } from 'react'
import { AlertCircle, Moon, Sun } from 'lucide-react'
import { api, type Config } from './api'
import Planner from './pages/Planner'

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
    try { localStorage.setItem('theme', dark ? 'dark' : 'light') } catch { /* private mode */ }
  }, [dark])
  return [dark, setDark] as const
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <rect x="8" y="10" width="14" height="13" rx="2.5" fill="none" stroke="#fff" strokeWidth="2.2" />
      <rect x="22" y="14" width="2.5" height="5" rx="1" fill="#fff" />
      <path d="M16.5 12.5 12.5 17h3l-1 3.5 4-4.5h-3z" fill="#fff" />
    </svg>
  )
}

export default function App() {
  const [dark, setDark] = useTheme()
  const [config, setConfig] = useState<Config | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { api.config().then(setConfig).catch((e) => setError(e.message)) }, [])

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface/85 backdrop-blur">
        <div className="mx-auto flex h-[60px] max-w-[1440px] items-center gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <Logo />
            <div className="leading-tight">
              <div className="text-[15px] font-semibold text-ink">Smart Battery Optimizer</div>
              <div className="hidden text-[11px] text-muted sm:block">Solar + storage scheduling for Pakistani homes</div>
            </div>
          </div>
          <button type="button" onClick={() => setDark(!dark)} aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
            className="ml-auto grid size-9 place-items-center rounded-lg border border-line text-ink-2 hover:bg-surface-2">
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
        </div>
      </header>

      {error ? (
        <div className="mx-auto max-w-[720px] px-4 py-16">
          <div role="alert" className="rounded-xl border border-line bg-surface p-6">
            <div className="flex items-center gap-2 text-[15px] font-semibold text-bad"><AlertCircle className="size-4" />Service unavailable</div>
            <p className="mt-2 text-[14px] text-ink-2">The planner could not be loaded. Please refresh the page or try again in a moment.</p>
          </div>
        </div>
      ) : config ? <Planner config={config} /> : (
        <div className="mx-auto max-w-[1440px] px-6 py-6"><div className="h-[600px] animate-pulse rounded-xl bg-surface-2" /></div>
      )}
    </div>
  )
}
