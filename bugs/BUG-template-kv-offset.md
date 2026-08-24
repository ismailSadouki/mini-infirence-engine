# KV-cache decode numerical divergence
Status: Known limitation / unresolved for M1.4

Observed:
- K/V cache contents match Hugging Face.
- Prefill logits match exactly.
- First decode token matches.
- Decode hidden states show BF16 numerical differences.
- Logits show small numerical differences.
- Over multiple decode steps, greedy argmax eventually diverges.
- 6/20 correctness prompts currently fail strict token identity.

Example:
Reference: [271, 40, 1079, 4460, 311, 1855, 264, 2025, 429, 646]
Cached:   [271, 40, 1079, 4460, 311, 1855, 264, 4285, 2025, 429]

Interpretation:
No evidence currently indicates incorrect KV storage.
The remaining difference appears during BF16 decode computation.