"""Provider adapter contract + shared behavior.

`decode` and `classify_error` are identical across hosts (they dispatch on the
capability's response shape / the exception), so they live here. Subclasses
supply `build_client`, `encode`, and `call` — the parts that are genuinely
provider-specific.
"""

from __future__ import annotations

import os
from typing import Any

from bench_core.capabilities import Capabilities
from bench_core.decode import decode_response
from bench_core.errors import Classification, classify
from bench_core.request import Request
from bench_core.response import Response


class ProviderAdapter:
    name: str
    base_url: str | None
    key_spec: list[str]  # env-var precedence, incl. lowercase fallbacks
    capability_defaults: dict

    def resolve_key(self) -> str | None:
        for env in self.key_spec:
            val = os.getenv(env)
            if val:
                return val
        return None

    # --- provider-specific (override) ---
    def build_client(self, api_key: str | None = None) -> Any:  # pragma: no cover
        raise NotImplementedError

    def encode(
        self, req: Request, caps: Capabilities, model_id: str
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    def call(self, client: Any, encoded: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    # --- shared ---
    def decode(self, raw: Any, caps: Capabilities) -> Response:
        return decode_response(raw, caps.response)

    def classify_error(self, exc: Exception) -> Classification:
        return classify(exc)
