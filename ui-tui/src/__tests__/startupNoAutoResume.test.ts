// display.tui_auto_resume_recent must not swap a fresh-intent launch (typed creation flags, `-w`, a
// dashboard workspace: the launcher sets HERMES_TUI_NO_AUTO_RESUME) for the most recent session (t_3d966055).
import { afterEach, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetUiState } from '../app/uiStore.js'

const startup = vi.hoisted(() => ({ noAutoResume: false }))
vi.mock('../config/env.js', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  get STARTUP_NO_AUTO_RESUME() {
    return startup.noAutoResume
  }
}))

afterEach(() => {
  startup.noAutoResume = false
  resetUiState()
})

const ready = () => {
  const newSession = vi.fn()
  const resumeById = vi.fn()

  const rpc = vi.fn(async (method: string) => {
    if (method === 'config.get') {
      return { config: { display: { tui_auto_resume_recent: true } } }
    }

    return method === 'session.most_recent' ? { session_id: 'sess-most-recent' } : null
  })

  createGatewayEventHandler({
    composer: { setInput: vi.fn(), enqueue: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession, resetSession: vi.fn(), resumeById, setCatalog: vi.fn() },
    submission: { submitLiteralRef: { current: vi.fn() }, submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, bellOnPrompt: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  } as any)({ payload: {}, type: 'gateway.ready' } as any)

  return { newSession, resumeById, rpc }
}

it('a fresh-intent launch forges a new session even with auto_resume on, never asking for the most recent', async () => {
  startup.noAutoResume = true
  const { newSession, resumeById, rpc } = ready()

  await vi.waitFor(() => expect(newSession).toHaveBeenCalled())
  expect(resumeById).not.toHaveBeenCalled()
  expect(rpc).not.toHaveBeenCalledWith('session.most_recent', expect.anything())
})

it('a bare launch still auto-resumes the most recent session', async () => {
  const { newSession, resumeById } = ready()

  await vi.waitFor(() => expect(resumeById).toHaveBeenCalledWith('sess-most-recent'))
  expect(newSession).not.toHaveBeenCalled()
})
