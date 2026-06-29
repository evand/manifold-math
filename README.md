# manifold-math

Math, machine-checked proofs, and reference implementations behind contributions to
[Manifold Markets](https://github.com/manifoldmarkets/manifold). Each subdirectory is a
self-contained project: a paper / write-up, executable proofs, and a Python reference
implementation that the upstream TypeScript can be checked against.

## Projects

- **[`cpmm-multi-2/`](cpmm-multi-2/)** — a second multi-choice market mechanism giving each
  answer its own `p` (non-uniform init + lossless liquidity add), reversible-limit auto-arb
  (`C(δ)+C(−δ)=0`), and a direct-formula `O(n log N)` cost core. Backs the upstream
  `cpmm-multi-2` PR. Paper + GP1–GP18 proofs + reference library + a differential anchor that
  reproduces a real captured Manifold bet to ~`1e-13`.

## License

MIT — see [LICENSE](LICENSE).
