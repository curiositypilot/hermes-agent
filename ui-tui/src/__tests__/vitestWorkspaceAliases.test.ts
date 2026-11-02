import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { workspaceAliases } from '../../vitest.config.js'

// Invariant under test: vitest resolves @hermes/ink and @hermes/shared to THIS
// checkout's source. Through node_modules a git worktree reaches the main
// checkout, whose hermes-ink has no built dist and whose apps/shared is another
// revision, so a worktree either failed to load or silently tested main's code.

const uiTui = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const repoRoot = dirname(uiTui)

const sharedExports = (
  JSON.parse(readFileSync(join(repoRoot, 'apps', 'shared', 'package.json'), 'utf8')) as {
    exports: Record<string, string>
  }
).exports

const matching = (specifier: string) => workspaceAliases.filter(alias => alias.find.test(specifier))

describe('vitest workspace aliases', () => {
  it('maps @hermes/ink to this checkout’s hermes-ink source entry', () => {
    const hits = matching('@hermes/ink')
    const entry = join(uiTui, 'packages', 'hermes-ink', 'src', 'entry-exports.ts')

    expect(hits.map(alias => alias.replacement)).toEqual([entry])
    expect(existsSync(entry)).toBe(true)
  })

  it('maps @hermes/ink/text-input to its own file, not the entry-exports alias', () => {
    const hits = matching('@hermes/ink/text-input')

    expect(hits.map(alias => alias.replacement)).toEqual([join(uiTui, 'packages', 'hermes-ink', 'text-input.js')])
    expect(existsSync(hits[0]!.replacement)).toBe(true)
  })

  it('maps every apps/shared export to exactly one existing file of this checkout', () => {
    const keys = Object.keys(sharedExports)

    expect(keys.length).toBeGreaterThan(1)

    for (const key of keys) {
      const specifier = `@hermes/shared${key.slice(1)}`
      const hits = matching(specifier)

      expect(hits, specifier).toHaveLength(1)
      expect(hits[0]!.replacement, specifier).toBe(join(repoRoot, 'apps', 'shared', sharedExports[key]!))
      expect(existsSync(hits[0]!.replacement), specifier).toBe(true)
    }

    expect(matching('@hermes/shared/billing').map(alias => alias.replacement)).toEqual([
      join(repoRoot, 'apps', 'shared', 'src', 'billing-types.ts')
    ])
  })

  it('points every alias inside this checkout', () => {
    for (const alias of workspaceAliases) {
      expect(alias.replacement.startsWith(repoRoot + sep), alias.replacement).toBe(true)
    }
  })

  it.each(['@hermes/shared-extra', '@hermes/shared/ansi/x', '@hermes/inkx', '@hermes/ink/', 'x@hermes/ink'])(
    'matches no alias for %s',
    specifier => {
      expect(matching(specifier)).toEqual([])
    }
  )
})
