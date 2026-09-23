# Lessons

Findings that cost real effort to establish and would otherwise be re-learned. Prose only — one
file per lesson, a claim and the measured numbers behind it. No code: scripts rot, and these are
meant to be read years from now.

Inclusion test: a conclusion that changes what someone does. Open questions go in issues, design
drafts in `dev/`, exploratory runs in `sandbox/` (both gitignored).

| lesson | takeaway |
|:--|:--|
| [sbi-summary-statistics](sbi-summary-statistics.md) | SBI samples `p(θ\|s)`, never `p(θ\|y)`. Insufficient statistics bias every sampler identically, so agreement between likelihood-free methods checks the samplers, not the inference. `var`/`cor` silently discard a known μ=0. |
