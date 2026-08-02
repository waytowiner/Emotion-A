"""弹性 HTTP 传输层。

本机环境特征（2026-07 实测）：
- 东财 push2（clist 列表接口）对 Python TLS 指纹/请求频率敏感，
  会间歇性 RST 连接；curl 与浏览器不受影响。
- push2his（K线接口）与 datacenter-web 相对稳定。
- 本地系统代理 (127.0.0.1:7890) 对东财同样不稳定，默认直连。

策略：curl_cffi 模拟 Chrome 指纹优先，requests 兜底；
指数退避 + 抖动重试；连接被重置视为可重试错误。
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from typing import Any
from urllib.parse import urlencode

import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ProxyError,
    requests.exceptions.ChunkedEncodingError,
)


class FetchError(RuntimeError):
    """所有重试均失败后抛出。"""


def _get_curl_cffi():
    try:
        from curl_cffi import requests as cr  # type: ignore

        return cr
    except Exception:
        return None


def _curl_get_json(url: str, params: dict[str, Any] | None, timeout: int) -> dict:
    """经系统 curl 二进制请求（Git Bash curl / OpenSSL，实测过东财 WAF 成功率最高）。"""
    full = f"{url}?{urlencode(params)}" if params else url
    out = subprocess.run(
        [
            "curl", "-s", "-m", str(timeout), "--noproxy", "*",
            "-H", f"User-Agent: {UA}",
            "-H", "Referer: https://quote.eastmoney.com/",
            "-H", "Accept: application/json, text/plain, */*",
            full,
        ],
        capture_output=True, timeout=timeout + 5,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise FetchError(f"curl rc={out.returncode} {out.stderr.decode(errors='ignore')[:120]}")
    return json.loads(out.stdout.decode("utf-8", errors="ignore"))


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    max_retries: int = 6,
    backoff_base: float = 2.0,
    use_proxy: bool = False,
) -> dict:
    """GET JSON，带弹性重试。优先 curl_cffi（Chrome 指纹），失败回退 requests。"""

    cr = _get_curl_cffi()
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        # 通道 1：系统 curl（WAF 通过率最高）
        try:
            return _curl_get_json(url, params, timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        # 通道 2：curl_cffi（Chrome 指纹）
        try:
            if cr is not None:
                resp = cr.get(
                    url,
                    params=params,
                    impersonate="chrome",
                    timeout=timeout,
                    proxies=None if not use_proxy else {},
                )
                if resp.status_code == 200 and resp.text.strip():
                    return resp.json()
                raise FetchError(f"HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:  # noqa: BLE001 - 传输层需要宽捕获
            last_exc = exc
        # 通道 3：requests 兜底
        try:
            s = requests.Session()
            s.trust_env = use_proxy  # False=忽略系统代理直连
            r = s.get(
                url,
                params=params,
                headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"},
                timeout=timeout,
            )
            if r.status_code == 200 and r.text.strip():
                return r.json()
            last_exc = FetchError(f"HTTP {r.status_code}: {r.text[:120]}")
        except _RETRYABLE as exc2:
            last_exc = exc2

        sleep_s = backoff_base**attempt * random.uniform(0.5, 1.5)
        time.sleep(min(sleep_s, 60))

    raise FetchError(f"fetch failed after {max_retries} tries: {url}") from last_exc
