# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |
| < 1.0 (beta) | No, please upgrade |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/fernandogarzaaa/callinsights/security/advisories/new)
or by email to **fernandogarzaaa@gmail.com**. Include a description, steps to reproduce,
and the affected version or commit. You should receive an acknowledgement within a few days,
and a fix or mitigation plan once the report is confirmed.

## Security model and hardening notes

- **Secrets**: all secrets come from environment variables. `ENV=production` refuses to
  start with a missing/short `SECRET_KEY` or a well-known `ADMIN_PASSWORD`. Never commit `.env`.
- **Authentication**: bcrypt password hashes; HS256 JWT sessions in an HttpOnly, SameSite=Lax
  cookie (Secure in production) or as a bearer token; API keys are random 256-bit tokens
  stored only as SHA-256 hashes and can be revoked.
- **CSRF**: cookie-authenticated writes must carry a same-origin `Origin`/`Referer`
  (extra origins via `TRUSTED_ORIGINS`). Bearer/API-key requests are not cookie-based.
- **Brute force**: login attempts are rate limited per client IP and email. The limiter is
  per process; add proxy-level limits for multi-instance deployments.
- **Uploads**: extension allow-list, streamed to disk under random names, size limits
  (`MAX_UPLOAD_MB`), and audio paths are always resolved inside the upload directory.
- **Output**: Jinja2 autoescaping, a restrictive Content-Security-Policy (no third-party
  scripts; UI assets are vendored), `X-Frame-Options: DENY`, `nosniff`, HSTS in production,
  and CSV exports neutralise spreadsheet formula injection.
- **Third parties**: call transcripts are sent to the LLM / ASR endpoint you configure. Use a
  local model (Ollama, faster-whisper) when recordings must not leave your infrastructure.
- **Container**: runs as an unprivileged user with hash-pinned dependencies.
