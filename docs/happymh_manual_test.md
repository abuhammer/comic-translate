# HappyMH Importer Manual Test

## Scenario
- Environment: containerized Ubuntu 24.04 image with PySide6 runtime dependencies (`libgl1`, `libegl1`, `libxkbcommon0`).
- Target chapter URL: `https://m.happymh.com/mangaread/yIjM3ATM4ATM=kTOxATM5ATM3QTM=YTOycTM2ATM2ATMzATMwUTM2UTM5ATM0ATM3ATM2QTM3UTM`.

## Steps
1. Installed the missing system libraries required for importing `PySide6`.
2. Spawned a temporary subclass of `ImportHappymhDownloadWorker` that overrides `_sleep_before_request` to eliminate the randomized throttling delay so the test can complete quickly in CI.
3. Executed the worker against the target URL inside a temporary directory.

## Result
- The importer reported zero errors, downloaded 76 images, and emitted progress events for the chapter, demonstrating that the flow no longer encounters HTTP 403 responses for this chapter.
