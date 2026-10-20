"""Environment-variable access with forgiving parsing.

The same variables reach the app by several routes — python-dotenv reading .env,
`docker run --env-file`, docker compose, a hosting dashboard — and these disagree
on syntax. In particular `docker run --env-file` keeps surrounding quotes and
trailing `# comments` as part of the value, so `RATE_LIMIT=10/minute  # per IP` or
`GOOGLE_API_KEY="abc"` would otherwise arrive as garbage. Normalising here means
every route behaves the same.
"""
import os
import re

_INLINE_COMMENT = re.compile(r"(^|\s+)#.*$")


def env_str(name: str, default: str = "") -> str:
    """The variable's value with whitespace, a trailing ` # comment` and one pair of
    surrounding quotes removed. Unset or blank values return `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = _INLINE_COMMENT.sub("", raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value or default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default
