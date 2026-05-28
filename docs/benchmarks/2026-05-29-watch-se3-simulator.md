# Watch SE 3 Simulator Validation

Date: 2026-05-29
Host Xcode: Xcode 26.3, build 17C529
Simulator runtime: watchOS 26.2
Destination: Apple Watch SE 3 (44mm)
Branch: `codex/watch-se-minicpm-foundation`

## Scope

This run validates that the current `WatchLMCore` Swift package builds and tests under the watchOS simulator target. It does not measure MiniCPM5-1B, Core ML, Neural Engine, real Apple Watch memory pressure, or thermal behavior.

Apple Watch simulator execution uses the host Mac, so these numbers are only useful for checking watchOS compatibility and current Swift contract overhead.

## Commands

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild build -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'
swift test
node --test
```

## Results

| Check | Result |
| --- | --- |
| watchOS simulator build | Passed |
| watchOS simulator Swift tests | 14 passed, 0 failed |
| macOS Swift tests | 14 passed, 0 failed |
| Node host tests | 37 passed, 0 failed |

Simulator benchmark print:

```text
WATCHLM_SIM_BENCH mock_short_turn iterations=1000 elapsed_ms=16.851 turns_per_second=59340.48
```

## Findings

- `WatchLMCore` compiles for `arm64-apple-watchos10.0-simulator`.
- Swift contracts and mock streaming runtime execute inside the Apple Watch SE 3 simulator.
- The test fixture loader had to be made simulator-safe because a watchOS simulator test bundle cannot read repository-relative `fixtures/` paths.
- The mock runtime overhead is tiny on simulator, but this does not predict MiniCPM decode speed.

## Real Performance Gap

The next meaningful performance test requires a real Core ML artifact:

- `prefill` `.mlpackage`
- one-token `decode` `.mlpackage`
- installed model manifest
- device or simulator benchmark report using `RuntimeTiming`

For the user's target, final speed claims must be collected on physical Apple Watch SE hardware, because simulator results do not model S8/S10, Neural Engine scheduling, watch memory limits, or thermal throttling.
