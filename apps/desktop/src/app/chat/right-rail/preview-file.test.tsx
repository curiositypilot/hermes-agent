import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setAlwaysExternalLinks } from '@/store/external-links'
import { $previewTabs } from '@/store/preview'
import { $dirtyPreviewUrls } from '@/store/preview-edit'

import { LocalFilePreview, MarkdownPreview } from './preview-file'

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders block and inline math through KaTeX', () => {
    // KaTeX marks its output; raw "$" delimiters must be gone.
    const { container } = render(
      <MarkdownPreview
        text={'Formula:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\n$$\n\nInline $a^2 + b^2 = c^2$ too.'}
      />
    )

    expect(container.querySelector('.katex')).not.toBeNull()
    expect(screen.queryByText(/\$\$/)).toBeNull()
  })

  it('renders GFM tables with header and body cells', () => {
    const { container } = render(<MarkdownPreview text={'| h1 | h2 |\n| --- | --- |\n| a | b |'} />)

    const table = container.querySelector('table')
    expect(table).not.toBeNull()
    expect(table?.querySelector('thead th')?.textContent).toBe('h1')
    expect(table?.querySelector('tbody td')?.textContent).toBe('a')
  })

  it('renders images with alt text', () => {
    const { container } = render(<MarkdownPreview text={'![a chart](https://example.com/chart.png)'} />)

    const img = container.querySelector('img')
    expect(img?.getAttribute('alt')).toBe('a chart')
    expect(img?.getAttribute('src')).toBe('https://example.com/chart.png')
  })

  it('opens an https link in the in-app browser instead of a blank window', async () => {
    $previewTabs.set([])
    setAlwaysExternalLinks(false)
    render(<MarkdownPreview text={'[docs](https://example.com/docs)'} />)

    const anchor = screen.getByRole('link', { name: 'docs' })
    const click = new MouseEvent('click', { bubbles: true, cancelable: true })

    anchor.dispatchEvent(click)
    expect(click.defaultPrevented).toBe(true)

    await waitFor(() => {
      expect(
        $previewTabs.get().some(tab => tab.target.kind === 'url' && tab.target.url === 'https://example.com/docs')
      ).toBe(true)
    })
  })

  it('scrolls a table-of-contents link to its heading without changing the app route', () => {
    $previewTabs.set([])

    const hash = window.location.hash
    const scroll = vi.fn()

    HTMLElement.prototype.scrollIntoView = scroll

    const { container } = render(
      <MarkdownPreview text={'## Managed Tiered KV Cache\n\n[jump](#managed-tiered-kv-cache)'} />
    )

    const heading = container.querySelector('#managed-tiered-kv-cache')

    expect(heading?.tagName).toBe('H2')
    fireEvent.click(screen.getByRole('link', { name: 'jump' }))
    expect(scroll).toHaveBeenCalledTimes(1)
    expect(window.location.hash).toBe(hash)
    expect($previewTabs.get()).toEqual([])
  })

  it('scrolls to a heading whose TOC link uses a different Unicode normalization', () => {
    const scroll = vi.fn()

    HTMLElement.prototype.scrollIntoView = scroll

    render(<MarkdownPreview text={'## \u30D3\n\n[jump](#\u30D2\u3099)'} />)

    fireEvent.click(screen.getByRole('link', { name: 'jump' }))
    expect(scroll).toHaveBeenCalledTimes(1)
  })

  it('opens a relative markdown link as the sibling file', async () => {
    $previewTabs.set([])
    render(<MarkdownPreview filePath="/vault/notes/index.md" text={'[next](../other.md)'} />)

    fireEvent.click(screen.getByRole('link', { name: 'next' }))

    await waitFor(() => {
      expect($previewTabs.get().some(tab => tab.target.kind === 'file' && tab.target.path === '/vault/other.md')).toBe(
        true
      )
    })
  })

  it('does not navigate a scriptable link', () => {
    $previewTabs.set([])

    const { container } = render(<MarkdownPreview text={'[bad](javascript:alert(1))'} />)

    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect($previewTabs.get()).toEqual([])
  })
})

// Spot editor save: only the snapshot actually written becomes the saved
// baseline. Keystrokes typed while the write is pending must stay dirty and
// in the editor, not be marked clean and discarded by the post-save reload.
describe('LocalFilePreview save', () => {
  const url = 'file:///workspace/note.txt'

  function setup() {
    let disk = 'original'
    let releaseWrite: (() => void) | null = null
    const writes: string[] = []

    vi.stubGlobal('hermesDesktop', {
      readFileText: async () => ({ binary: false, path: '/workspace/note.txt', text: disk, truncated: false }),
      writeTextFile: (_file: string, content: string) => {
        writes.push(content)

        return new Promise<{ path: string }>(resolve => {
          releaseWrite = () => {
            disk = content
            resolve({ path: '/workspace/note.txt' })
          }
        })
      }
    })

    const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)

    Object.defineProperties(Range.prototype, {
      getClientRects: { configurable: true, value: () => [] },
      getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
    })

    const restore = () => {
      for (const key of ['getClientRects', 'getBoundingClientRect']) {
        if (rangeDescriptors[key]) {
          Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
        } else {
          Reflect.deleteProperty(Range.prototype, key)
        }
      }
    }

    const rendered = render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'note.txt',
          path: '/workspace/note.txt',
          previewKind: 'text',
          source: '/workspace/note.txt',
          url
        }}
      />
    )

    return {
      disk: () => disk,
      async edit() {
        fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))

        return EditorView.findFromDOM(rendered.container.querySelector('.cm-content')!)!
      },
      async finishWrite() {
        await waitFor(() => expect(releaseWrite).not.toBeNull())
        await act(async () => {
          releaseWrite!()
          releaseWrite = null
        })
      },
      restore,
      writes
    }
  }

  const type = (editor: EditorView, text: string) =>
    act(() => editor.dispatch({ changes: { from: editor.state.doc.length, insert: text } }))

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    $dirtyPreviewUrls.set({})
  })

  it('keeps keystrokes typed during an in-flight save dirty and in the editor', async () => {
    const harness = setup()

    try {
      const editor = await harness.edit()
      type(editor, ' first')
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await waitFor(() => expect(harness.writes).toEqual(['original first']))
      type(editor, ' second')
      await harness.finishWrite()

      expect(harness.disk()).toBe('original first')
      expect(editor.dom.isConnected).toBe(true)
      expect(editor.state.doc.toString()).toBe('original first second')
      expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull()
      await waitFor(() =>
        expect((screen.getByRole('button', { name: /Save/ }) as HTMLButtonElement).disabled).toBe(false)
      )
      expect($dirtyPreviewUrls.get()[url]).toBe(true)

      // The baseline is the written snapshot, so the next Save passes the
      // stale-on-disk check (no conflict banner) and writes the rest.
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await waitFor(() => expect(harness.writes).toEqual(['original first', 'original first second']))
      await harness.finishWrite()
      expect(screen.queryByRole('button', { name: /Overwrite/ })).toBeNull()
      await screen.findByRole('button', { name: 'Edit' })
      expect(harness.disk()).toBe('original first second')
      expect($dirtyPreviewUrls.get()[url]).toBeUndefined()
    } finally {
      harness.restore()
    }
  })

  it('keeps a revert to the old baseline typed during an in-flight save dirty and savable', async () => {
    const harness = setup()

    try {
      const editor = await harness.edit()
      type(editor, ' first')
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await waitFor(() => expect(harness.writes).toEqual(['original first']))
      // Delete ' first' while the write pends: the draft now equals the OLD
      // baseline, which the per-keystroke check reads as clean.
      act(() => editor.dispatch({ changes: { from: 'original'.length, to: editor.state.doc.length } }))
      await harness.finishWrite()

      expect(harness.disk()).toBe('original first')
      expect(editor.dom.isConnected).toBe(true)
      expect(editor.state.doc.toString()).toBe('original')
      await waitFor(() =>
        expect((screen.getByRole('button', { name: /Save/ }) as HTMLButtonElement).disabled).toBe(false)
      )
      expect($dirtyPreviewUrls.get()[url]).toBe(true)

      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await waitFor(() => expect(harness.writes).toEqual(['original first', 'original']))
      await harness.finishWrite()
      expect(screen.queryByRole('button', { name: /Overwrite/ })).toBeNull()
      await screen.findByRole('button', { name: 'Edit' })
      expect(harness.disk()).toBe('original')
      expect($dirtyPreviewUrls.get()[url]).toBeUndefined()
    } finally {
      harness.restore()
    }
  })

  it('marks the draft clean and leaves edit mode when nothing was typed during the save', async () => {
    const harness = setup()

    try {
      const editor = await harness.edit()
      type(editor, ' first')
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await harness.finishWrite()

      await screen.findByRole('button', { name: 'Edit' })
      expect(harness.disk()).toBe('original first')
      expect(harness.writes).toEqual(['original first'])
      expect($dirtyPreviewUrls.get()[url]).toBeUndefined()
    } finally {
      harness.restore()
    }
  })
})
