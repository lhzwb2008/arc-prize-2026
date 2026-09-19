#!/usr/bin/env python3
"""Refresh GPU2 Kaggle OAuth if the access token expires within 2 hours.

Kaggle issues 12h access tokens. leftover_8x8 submit is 2026-09-20T00:00:00Z
and the watch cron keeps polling for ~12h after that, so a token minted at
push time will die mid-window. Cron should call this before submit/watch.
Does not print tokens.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials

    now = datetime.now(timezone.utc)
    api = KaggleApi()
    api._load_config()
    with api.build_kaggle_client() as client:
        creds = KaggleCredentials.load(client=client)
        if not creds:
            print(f"[{now.strftime('%Y-%m-%dT%H:%M:%SZ')}] no oauth creds; skip", flush=True)
            return 0
        exp = creds._access_token_expiration
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp is not None and exp > now + timedelta(hours=2):
            print(
                f"[{now.strftime('%Y-%m-%dT%H:%M:%SZ')}] oauth ok until {exp.isoformat()}",
                flush=True,
            )
            return 0
        creds.refresh_access_token()
        print(
            f"[{now.strftime('%Y-%m-%dT%H:%M:%SZ')}] refreshed oauth until "
            f"{creds._access_token_expiration.isoformat()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
