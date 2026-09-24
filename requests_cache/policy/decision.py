"""Optional tracing of cache decisions for a single request.

When enabled (``CachedSession(cache_trace=True)``), a :py:class:`.CacheDecision` is built during
request handling and attached to the returned response as ``response.cache_decision``. It records
the inputs and outputs of each decision point (cache key composition, expiration source, cache
directives, staleness checks, conditional headers, and the final action taken) so callers can
understand *why* a response was (or wasn't) served from the cache.

The trace only observes values that were already determined while handling the current request; it
does not read the clock or the cache backend on its own. Sensitive data is excluded: request bodies
and sensitive headers (e.g. ``Authorization``) never appear in the trace.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from attrs import define, field

from ..cache_keys import get_matched_headers, normalize_request
from ..models import RichMixin
from . import ExpirationPattern, ExpirationTime, HeaderDict
from .settings import AUTH_HEADERS

if TYPE_CHECKING:
    from requests import PreparedRequest

    from .directives import CacheDirectives
    from .settings import CacheSettings

__all__ = [
    'ACTION_ERROR_504',
    'ACTION_HIT',
    'ACTION_RESEND',
    'ACTION_REVALIDATED',
    'ACTION_SEND',
    'ACTION_STALE_IF_ERROR',
    'ACTION_STALE_WHILE_REVALIDATE',
    'CacheDecision',
    'CacheGeneration',
    'KeySummary',
    'VaryCheck',
    'redact_headers',
    'summarize_key',
    'url_rule_name',
]

# Final actions a request can resolve to
ACTION_HIT = 'hit'  # Fresh cached response returned
ACTION_SEND = 'send'  # New request sent (cache miss, or cache read skipped)
ACTION_RESEND = 'resend'  # Stale cached response; new request sent and returned
ACTION_REVALIDATED = 'revalidated'  # Conditional request returned 304; cached response updated
ACTION_STALE_IF_ERROR = 'stale_if_error'  # Request failed; stale cached response returned
ACTION_STALE_WHILE_REVALIDATE = (
    'stale_while_revalidate'  # Stale response returned; refreshing in background
)
ACTION_ERROR_504 = 'error_504'  # only-if-cached requested, but no usable cached response


@define(repr=False)
class KeySummary(RichMixin):
    """Summary of the components used to build a cache key. Contains no header values or body
    content; only metadata about what contributed to the key.
    """

    method: str = field(default=None)
    url: str = field(default=None)
    body_in_key: bool = field(default=False)
    body_size: int = field(default=0)
    matched_headers: List[str] = field(factory=list)
    ignored_parameters: List[str] = field(factory=list)
    serializer: str = field(default=None)


@define(repr=False)
class VaryCheck(RichMixin):
    """Result of validating a cached response's ``Vary`` header against the current request"""

    header: Optional[str] = field(default=None)
    matched: Optional[bool] = field(default=None)
    vary_cache_key: Optional[str] = field(default=None)
    secondary_hit: Optional[bool] = field(default=None)


@define(repr=False)
class CacheGeneration(RichMixin):
    """One generation of a cached response. After a 304 revalidation, the returned response links
    two generations: the old one (which provides the body) and the new one (which provides updated
    headers and expiration).
    """

    source: str = field(default=None)  # 'stored' or 'origin'
    status_code: int = field(default=None)
    created_at: Optional[datetime] = field(default=None)
    expires: Optional[datetime] = field(default=None)
    headers: Dict[str, str] = field(factory=dict)
    body_size: int = field(default=0)


@define(repr=False)
class CacheDecision(RichMixin):
    """A trace of how the cache handled a single request. All values are recorded at the decision
    points where they were determined, for the same request; nothing is re-read from the clock or
    the cache backend after the fact.
    """

    # Final outcome; one of the ACTION_* constants
    action: Optional[str] = field(default=None)

    # Cache key and its composition
    cache_key: str = field(default=None)
    key_summary: Optional[KeySummary] = field(default=None)

    # Expiration inputs
    expire_after: ExpirationTime = field(default=None)
    expiration_source: Optional[str] = field(default=None)  # request|url_rule|session|response
    url_rule: Optional[str] = field(default=None)

    # Cache directives parsed from request and response headers
    request_directives: Optional['CacheDirectives'] = field(default=None)
    response_directives: Optional['CacheDirectives'] = field(default=None)

    # State of the stored response at lookup time
    backend_hit: Optional[bool] = field(default=None)
    stored_created_at: Optional[datetime] = field(default=None)
    stored_expires: Optional[datetime] = field(default=None)
    stored_age: Optional[timedelta] = field(default=None)
    now: Optional[datetime] = field(default=None)

    # Feature-specific decision nodes
    vary: VaryCheck = field(factory=VaryCheck)
    only_if_cached: bool = field(default=False)
    max_stale: Optional[int] = field(default=None)
    min_fresh: Optional[int] = field(default=None)
    stale_if_error: Union[bool, ExpirationTime] = field(default=None)
    stale_while_revalidate: Union[bool, ExpirationTime] = field(default=None)
    refresh: bool = field(default=False)
    disabled: bool = field(default=False)

    # Conditional (validation) headers applied to an outgoing request
    conditional_headers: Dict[str, str] = field(factory=dict)

    # Reasons for skipping cache read/write, if any
    read_skip_reasons: List[str] = field(factory=list)
    write_skip_reasons: List[str] = field(factory=list)

    # Response generations linked by a 304 revalidation (old body + new headers)
    generations: List[CacheGeneration] = field(factory=list)


def summarize_key(
    request: 'PreparedRequest',
    settings: 'CacheSettings',
    serializer=None,
) -> KeySummary:
    """Summarize the components that make up a cache key, without recording sensitive values"""
    norm_request = normalize_request(
        request, settings.ignored_parameters, settings.content_root_key
    )
    matched = get_matched_headers(norm_request.headers, settings.match_headers)
    body = norm_request.body or b''
    return KeySummary(
        method=norm_request.method,
        url=norm_request.url,
        body_in_key=bool(body),
        body_size=len(body),
        matched_headers=[h.split('=', 1)[0] for h in matched],
        ignored_parameters=sorted(settings.ignored_parameters or []),
        serializer=getattr(serializer, 'name', None) or str(serializer),
    )


def redact_headers(headers: HeaderDict, ignored_parameters=()) -> Dict[str, str]:
    """Return a copy of headers with sensitive (auth and ignored) headers removed"""
    ignored = {h.lower() for h in (*AUTH_HEADERS, *(ignored_parameters or ()))}
    return {k: v for k, v in (headers or {}).items() if k.lower() not in ignored}


def url_rule_name(pattern: Optional[ExpirationPattern]) -> Optional[str]:
    """Get a displayable name for a URL expiration pattern (glob string or regex)"""
    if pattern is None:
        return None
    return getattr(pattern, 'pattern', None) or str(pattern)
