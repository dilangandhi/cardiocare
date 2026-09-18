"""Container health probe.

Kept in its own file rather than inlined into the Dockerfile's HEALTHCHECK.
An inline probe has to survive the Dockerfile parser, then /bin/sh, then
Python, and every layer has its own opinion about quotes and backslashes --
which is how a stray escape turned into a "missing end of string" build
failure. A file has no quoting at all.

Exit 0 means healthy; any exception exits non-zero.
"""

import os
import sys
import urllib.request

port = os.environ.get("PORT", "7860")
url = "http://localhost:" + port + "/api/health"

try:
    with urllib.request.urlopen(url, timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:  # noqa: BLE001
    print("health probe failed: " + str(exc), file=sys.stderr)
    sys.exit(1)
