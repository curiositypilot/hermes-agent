import { describe, expect, it } from 'vitest'

import { ENUM_OPTIONS } from './constants'

describe('ENUM_OPTIONS tts.kittentts.model', () => {
  // KittenML publishes int8 weights only for nano; micro and mini exist only as
  // full-precision repos. Offering a repo that does not exist makes TTS fail
  // the moment the user selects it (NousResearch/hermes-agent#135767).
  it('offers exactly the KittenTTS repos that exist on Hugging Face', () => {
    expect(ENUM_OPTIONS['tts.kittentts.model']).toEqual([
      'KittenML/kitten-tts-nano-0.8-int8',
      'KittenML/kitten-tts-micro-0.8',
      'KittenML/kitten-tts-mini-0.8'
    ])
  })

  it('never offers an int8 variant of micro or mini', () => {
    for (const id of ENUM_OPTIONS['tts.kittentts.model']) {
      expect(id).not.toMatch(/(micro|mini)-0\.8-int8$/)
    }
  })
})
