# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GLM-5.2 decode megakernel: all 78 decoder layers of one decode step in one pallas_call.

`kernel.py`, `layout.py` and `config.py` are vendored unchanged from the megakernel's
development tree (its TINY interpret/hardware oracle lives there); `vllm_adapter.py` is the
serving glue. Enabled by `USE_GLM5_MEGAKERNEL=1`.
"""
