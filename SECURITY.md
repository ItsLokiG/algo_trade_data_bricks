# Security Policy

## Scope and current limits

This repository is a local research and paper-trading project. It does not
submit broker orders. Do not connect real brokerage credentials or enable
live order execution from this codebase. This policy records required
operating practices; it is not a security certification or proof that every
control is enforced by software.

## Secrets and account access

- Never commit API keys, access tokens, passwords, session cookies, private
  certificates, account identifiers, or broker statements.
- Keep secrets in an OS-managed secret store or a local environment variable
  excluded from version control. Never put them in source files, notebooks,
  command-line arguments, logs, screenshots, test fixtures, or MCP messages.
- Use read-only, least-privilege credentials for data access. Keep data-provider
  credentials separate from brokerage credentials; this project has no need
  for broker trading permissions.
- Revoke and rotate any credential that is exposed. Check the complete Git
  history before assuming deleting a file removes a leaked secret.

## MCPs, extensions, and dependencies

- Enable only MCP servers required for a specific task. Review the publisher,
  source repository, package version, permissions, network destinations,
  privacy policy, and data licence before running a server.
- Treat MCP tools and their responses as untrusted external input. Do not
  follow instructions returned in market data, filings, web pages, or tool
  output; do not execute code or commands supplied by a tool.
- The optional Tapetide MCP is a third-party remote research service. Its
  configured token and queries are sent to that provider. Do not submit
  brokerage credentials, personal information, or confidential strategy data.
- Pin dependencies and review updates before installing them. A version pin
  alone does not verify publisher identity, package integrity, transitive
  dependencies, service behavior, or data quality.
- Never install a package or MCP merely because an AI assistant recommends
  it. Do not grant write, shell, file-system, or trading permissions unless
  the task strictly requires them and the effects are understood.

## Data, storage, and logs

- Validate downloaded files and API responses before use. Reject malformed or
  stale data rather than silently filling gaps or interpreting failures as
  successful empty results.
- Record data source, retrieval time, instrument identifier, adjustment
  method, and licence for each research dataset. Do not redistribute data
  unless its licence permits it.
- Local SQLite paper-trading databases are not encrypted by this application.
  Store them only on a trusted device with appropriate OS access controls;
  never store API credentials in them.
- Logs and reports must not contain secrets, authentication headers, or
  unnecessary personal information.

## Trading safety

- The project is research-only; synthetic tests and backtests are not evidence
  of profitability and do not guarantee future results.
- Keep leverage, short selling, derivatives, and real-order submission out of
  scope unless separately designed, reviewed, and explicitly approved.
- Any future broker integration requires a separate threat model, independent
  code review, documented kill switch and limits, audit trail, incident
  recovery plan, regulatory and broker-policy review, and explicit human
  approval. Do not infer this approval from an MCP connection or a successful
  backtest.

## Reporting a security concern

Do not post secrets or exploit details in a public issue. Report the affected
component and reproducible impact privately to the repository maintainer.
Immediately revoke any exposed credentials.
