import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { canonicalRequest, canonicalResult } from '../canonicalGateway.js'

// `hermes --tui --resume <id> -m X` (t_942f47e1): a typed creation flag the session was not launched with
// refuses the startup resume as classic chat does, instead of attaching on the frozen route without a word.

const startup = vi.hoisted(() => ({ check: null as null | Record<string, unknown> }))

vi.mock('../config/env.js', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  get STARTUP_RESUME_CHECK() {
    return startup.check
  },
  STARTUP_QUERY: 'startup prompt'
}))

function launch(home: string) {
  resetUiState()

  const request = vi.fn(async (method: string, original: Record<string, unknown> = {}) => {
    const wire = canonicalRequest(method, original)

    if (wire.method === 'session.resume') {
      return canonicalResult(method, {
        session_id: 'owner', stored_session_id: 'owner', subscription_id: 'sub-1', authority_epoch: 1,
        execution_generation: 0, messages: [{ role: 'user', content: 'earlier turn' }], pending: [], running: false,
        info: { model: 'frozen/model', cwd: home, launch_request: { model: 'frozen/model' }, tools: {}, skills: {} }
      })
    }

    return wire.method === 'session.detach' ? { ...wire.params, detached: true } : {}
  })

  const setHistoryItems = vi.fn()
  const submit = vi.fn()
  const dieWithCode = vi.fn()
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() }, gw: { isCanonical: true, request },
      rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(), setHistoryItems,
      setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn()
    } as any)

    return <Text>startup resume</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  const handler = createGatewayEventHandler({
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    composer: { setInput: vi.fn(), enqueue: vi.fn() },
    session: {
      STARTUP_RESUME_ID: 'owner', colsRef: { current: 80 }, dieWithCode, newSession: vi.fn(), resetSession: vi.fn(),
      resumeById: (id: string, opts?: any) => lifecycle.resumeById(id, opts), setCatalog: vi.fn()
    },
    submission: { submitLiteralRef: { current: submit }, submitRef: { current: vi.fn() } },
    system: { sys: vi.fn(), bellOnComplete: false, bellOnPrompt: false },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  } as any)

  handler({ type: 'gateway.ready', payload: {} } as any)

  const calls = (name: string) => request.mock.calls.filter(([method]) => method === name).map(([, params]) => params ?? {})

  return { calls, dieWithCode, setHistoryItems, submit, unmount: () => instance.unmount() }
}

it('refuses the startup resume on a flag the session was not launched with: no attach, no history, no prompt', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-resume-check-'))
  const refusal = join(home, 'refusal.json')
  writeFileSync(refusal, '')
  vi.stubEnv('HERMES_HOME', home)
  startup.check = { requested: { model: 'other/model', yolo: true }, model_resolved: 'other/model', refusal_file: refusal }
  const app = launch(home)

  try {
    await vi.waitFor(() => expect(app.dieWithCode).toHaveBeenCalledWith(2))
    expect(JSON.parse(readFileSync(refusal, 'utf8'))).toEqual(['model', 'yolo'])
    // The attach is released as it was taken: never main's tui_close / tui_shutdown stamp.
    expect(app.calls('session.detach')).toEqual([{ session_id: 'owner', subscription_id: 'sub-1' }])
    expect(app.calls('session.create')).toEqual([])
    expect(app.setHistoryItems).not.toHaveBeenCalled()
    expect(getUiState().sid).toBeNull()
    await new Promise(resolve => setTimeout(resolve, 150))
    expect(app.submit).not.toHaveBeenCalled()
  } finally {
    app.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})

it('resumes as before when the typed flags repeat the session launch', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-resume-check-'))
  const refusal = join(home, 'refusal.json')
  writeFileSync(refusal, '')
  vi.stubEnv('HERMES_HOME', home)
  startup.check = { requested: { model: 'frozen/model', cwd: home }, refusal_file: refusal }
  const app = launch(home)

  try {
    await vi.waitFor(() => expect(getUiState().sid).toBe('owner'))
    await vi.waitFor(() => expect(app.submit).toHaveBeenCalledWith('startup prompt', []))
    expect(app.dieWithCode).not.toHaveBeenCalled()
    expect(app.calls('session.detach')).toEqual([])
    expect(app.setHistoryItems).toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ text: 'earlier turn' })]))
    expect(readFileSync(refusal, 'utf8')).toBe('')
  } finally {
    app.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
