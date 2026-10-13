// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { PlatformAvatar } from './platform-icon'

afterEach(cleanup)

describe('PlatformAvatar', () => {
  // Matrix's mark is black by design; painting it #000 on its own 16% tint
  // made the chip invisible on dark themes. It must follow the text color.
  it('renders the Matrix glyph in the surrounding text color, not brand black', () => {
    const { container } = render(<PlatformAvatar platformId="matrix" platformName="Matrix" />)
    const chip = container.firstElementChild as HTMLElement

    expect(chip.querySelector('svg')).not.toBeNull()
    expect(chip.style.color).toBe('')
  })

  it('keeps brand color for colored marks', () => {
    const { container } = render(<PlatformAvatar platformId="telegram" platformName="Telegram" />)

    expect((container.firstElementChild as HTMLElement).style.color).not.toBe('')
  })
})
