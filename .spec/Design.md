# Master OpenDesign Contract (DESIGN.md)
## Global Design Tokens for Project SYZYGY

### 1. Color Palette (Dark Obsidian + Indigo / Emerald Accents)
- **Background Primary:** `#090D16` (Deep Space Obsidian)
- **Background Surface:** `#0F172A` (Slate 900)
- **Background Surface Hover:** `#1E293B` (Slate 800)
- **Accent Primary (Brand & Controls):** `#6366F1` (Indigo 500)
- **Accent Positive (Low Risk & Mitigators):** `#10B981` (Emerald 500)
- **Accent Warning (Medium Risk):** `#F59E0B` (Amber 500)
- **Accent Danger (High Risk & Escalators):** `#F43F5E` (Rose 500)
- **Text Primary:** `#F8FAFC` (Slate 50)
- **Text Secondary:** `#94A3B8` (Slate 400)
- **Text Tertiary:** `#64748B` (Slate 500)
- **Border Subtle:** `rgba(255, 255, 255, 0.08)`
- **Border Active:** `rgba(99, 102, 241, 0.35)`

### 2. Typography
- **Title Font:** `Outfit`, sans-serif (Weights: 600, 700, 800; Tracking: -0.025em)
- **Body Font:** `Inter`, sans-serif (Weights: 400, 500, 600)
- **Code & Tabular Font:** `JetBrains Mono`, monospace (font-variant-numeric: tabular-nums)

### 3. Layout Grid
- 8-pixel base grid: `8px`, `16px`, `24px`, `32px`, `48px`, `64px`

### 4. Motion Physics (Motion Division Standard)
- Spring curve: `cubic-bezier(0.23, 1, 0.32, 1)`
- Tactile feedback: `150ms` instant response (`transform: scale(0.97)` on `:active`)
