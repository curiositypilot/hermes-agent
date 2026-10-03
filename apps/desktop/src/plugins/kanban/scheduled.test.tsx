import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { TaskDrawer } from './drawer'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanTask } from './types'
import { scheduledWake, wakeFull, wakeShort } from './ui'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

// 2026-10-07 09:30 local — a non-midnight wake, so the chip carries a clock.
const WAKE = new Date(2026, 9, 7, 9, 30).getTime() / 1000

let task: KanbanTask
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string): Promise<unknown> => {
  if (path === '/tasks/t_example') {
    return {
      task,
      comments: [],
      events: [],
      attachments: [],
      links: { parents: [], children: [] },
      runs: []
    }
  }

  if (path.startsWith('/tasks/t_example/log?')) {
    return { exists: false, content: '', size_bytes: 0, truncated: false }
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  if (path === '/orchestration') {
    return { default_assignee: '' }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, _options?: PluginRestOptions) => (await rest(path)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['todo', 'scheduled', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
    </QueryClientProvider>
  )
}

describe('dated schedule display', () => {
  it('shows the wake date and the start outcome for a dated scheduled task', async () => {
    task = { id: 't_example', title: 'Dated', status: 'scheduled', scheduled_until: WAKE, scheduled_then: 'start' }
    openDrawer()

    await screen.findByRole('heading', { name: 'Dated' })
    const at = new Date(WAKE * 1000)
    expect(screen.getByText(en.metaWakes)).toBeTruthy()
    expect(screen.getByText(`${wakeFull(at)} → ${en.scheduleOutcome.start}`)).toBeTruthy()
  })

  it('shows no date for an undated scheduled task', async () => {
    task = { id: 't_example', title: 'Undated', status: 'scheduled', scheduled_until: null, scheduled_then: null }
    openDrawer()

    await screen.findByRole('heading', { name: 'Undated' })
    expect(screen.queryByText(en.metaWakes)).toBeNull()
    expect(scheduledWake(task)).toBeNull()
  })

  it('derives the card chip from status + columns, with the clock only off midnight', () => {
    const dated: KanbanTask = {
      id: 't',
      title: '',
      status: 'scheduled',
      scheduled_until: WAKE,
      scheduled_then: 'start'
    }

    expect(scheduledWake(dated)).toEqual({ at: new Date(WAKE * 1000), then: 'start' })
    // A stale date on a card that already woke is not a schedule.
    expect(scheduledWake({ ...dated, status: 'ready' })).toBeNull()
    // Missing mode reads as the backend default, ask.
    expect(scheduledWake({ ...dated, scheduled_then: null })?.then).toBe('ask')

    const midnight = new Date(2026, 9, 7)
    expect(wakeShort(new Date(WAKE * 1000)).length).toBeGreaterThan(wakeShort(midnight).length)
    expect(wakeShort(new Date(WAKE * 1000)).startsWith(wakeShort(midnight))).toBe(true)
  })
})
