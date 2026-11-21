# Loky Resource-Tracker `KeyError` During Windows Jupyter Shutdown

**Status:** Low priority — deferred  
**Identified:** PR 41, Commit 6B  
**Disposition:** Upstream compatibility issue; no repository changes required.

## Environment

- Windows 11
- Python 3.14.3
- Joblib 1.6.0 (vendored Loky 3.6.0)
- ipykernel 6.31.0

## Issue

After completing GLASS and black-box notebook execution, Joblib/Loky's resource tracker may emit:

```text
resource_tracker.py, line 386
del cache[rtype][name]
KeyError: '...\joblib_memmapping_folder_...'
```

Both notebooks previously completed **9/9 execution checks** and successfully generated their execution records.

## Root Cause

Local reproduction supports a shutdown-order conflict between ipykernel and Joblib/Loky.

During kernel shutdown, ipykernel terminates child processes, including Loky's resource tracker. Joblib's subsequent exit-time cleanup may restart the tracker with an empty registry. Attempts to unregister previously tracked temporary folders then produce `KeyError` tracebacks.

This occurs during kernel shutdown, after notebook execution has completed.

## Impact

No impact on notebook results, accepted artifacts, or execution records has been observed.

Local reproduction confirmed successful kernel execution, with no remaining Joblib temporary folders or orphaned tracker/worker processes.

The primary observed impact is a misleading traceback during otherwise successful execution.

## Verification

Reproduced locally using two standalone diagnostic scripts:

- **Isolated reproduction:** Normal shutdown completed cleanly; simulated ipykernel-style shutdown reproduced the traceback.
- **Real-kernel reproduction:** Confirmed tracker relaunch and the same `KeyError` during shutdown.

Both diagnostic scripts remain outside the repository.

## Future Resolution

Investigate [Joblib PR #1829](https://github.com/joblib/joblib/pull/1829), which addresses temporary-folder registration and cleanup behavior.

When a potentially compatible Joblib release becomes available, repeat the isolated reproduction before upgrading.

**Priority:** Low. Revisit in a future maintenance PR or GitHub issue if the problem persists.