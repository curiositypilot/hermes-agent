import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { defineConfig } from 'vitest/config'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))
const exact = (specifier: string) => new RegExp(`^${specifier.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')}$`)

const shared = JSON.parse(readFileSync(here('../apps/shared/package.json'), 'utf8')) as {
  exports: Record<string, string>
}

// Resolve the workspace packages to THIS checkout's source, as scripts/build/tui.mjs
// does for the bundle. Through node_modules a git worktree reaches the main checkout
// (its node_modules are symlinks), where @hermes/ink has no built dist and
// @hermes/shared is a different revision of the code under test.
export const workspaceAliases = [
  { find: exact('@hermes/ink'), replacement: here('./packages/hermes-ink/src/entry-exports.ts') },
  { find: exact('@hermes/ink/text-input'), replacement: here('./packages/hermes-ink/text-input.js') },
  ...Object.entries(shared.exports).map(([subpath, file]) => ({
    find: exact(`@hermes/shared${subpath.slice(1)}`),
    replacement: here(`../apps/shared/${file}`)
  }))
]

export default defineConfig({
  resolve: { alias: workspaceAliases },
  test: {
    exclude: ['dist/**', 'node_modules/**']
  }
})
