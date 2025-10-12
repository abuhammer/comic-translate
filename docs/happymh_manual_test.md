# HappyMH Importer Manual Test

## Scenario A (original reproduction URL)
- Environment: containerized Ubuntu 24.04 image with PySide6 runtime dependencies (`libgl1`, `libegl1`, `libxkbcommon0`).
- Target chapter URL: `https://m.happymh.com/mangaread/yIjM3ATM4ATM=kTOxATM5ATM3QTM=YTOycTM2ATM2ATMzATMwUTM2UTM5ATM0ATM3ATM2QTM3UTM`.

### Steps
1. Installed the missing system libraries required for importing `PySide6`.
2. Spawned a temporary subclass of `ImportHappymhDownloadWorker` that overrides `_sleep_before_request` to eliminate the randomized throttling delay so the test can complete quickly in CI.
3. Executed the worker against the target URL inside a temporary directory; the worker now auto-primes Cloudflare cookies by fetching the hashed `main.*.js` bundle when the reading page returns a human-verification challenge, ensuring the API version fallback stays up to date.

### Result
- The importer reported zero errors, downloaded 76 images, and emitted progress events for the chapter, demonstrating that the challenge-aware retry logic prevents HTTP 403 responses for this chapter.

## Scenario B (regression URL shared on follow-up)
- Environment: same as Scenario A.
- Target chapter URL: `https://m.happymh.com/mangaread/yIjM3ATM4ATM0ATM2ATMxETM0UTM1ATMycTM2ATM2ATMzATMwUTM2UTM5ATM0ATM3ATM2QTM3UTM`.

### Steps
1. Reused the installed system libraries so `PySide6` imports succeed.
2. Leveraged the importer helper methods (`_bootstrap_session`, `_fetch_reading_page`, `_request_chapter_data`, etc.) inside a temporary directory to simulate the worker's flow without persisting the downloaded scans.
3. Allowed the helper sequence to retry with cached and fallback API versions when necessary.

### Result
- The importer retrieved JSON metadata for the chapter without encountering a 403, reported the active API version (`v3.1818134`), and enumerated 72 scan URLs, confirming the challenge-aware logic also succeeds for the new reproduction link.
