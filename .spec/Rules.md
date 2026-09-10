# Agent & Code Directives (ASD-STE100 Standard)
## Project: SYZYGY Engineering Rules

1. **Deterministic Execution:** No placeholder mocks or simulated fake data in production inference code.
2. **Simplified Technical English (ASD-STE100):** All user-facing text must be direct, clear, and unambiguous. Eliminate AI buzzwords ("Next-Gen", "Seamless", "Unleash", "Game-changer").
3. **Spec-Driven Consistency:** Any structural or architectural changes must update `.spec/` before merging.
4. **Backward Compatibility:** Maintain legacy field aliases (e.g., `cost_savings_cr` and `cost_saved_cr`) to prevent breaking external API consumers.
5. **Performance Standards:** Single inference must complete within 50ms locally.
