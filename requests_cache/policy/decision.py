"""Optional cache decision tracing.

When :py:attr:`.CacheSettings.trace_decisions` is enabled, a :py:class:`CacheDecision` is built up
during :py:meth:`.CachedSession.send` and attached to the returned response as
``response.cache_decision``. It records the deterministic inputs and outputs of the cache policy
for a single request: cache key composition, matched URL expiration rule, request/response cache
directives, stored response age, conditional request headers, backend hits, and the final action.

The trace only observes values that are already computed while handling the request. It does not
re-read the system clock (a single timestamp is captured when the decision is created) and does
not perform any additional backend reads. Sensitive header values and request/response bodies are
never recorded; the request body is represented only by its hash.
"""

from datetime import datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attrs
from attrs import define, field

from .._utils import encode
from ..cache_keys import normalize_request
from ..models import RichMixin
from . import ExpirationPattern, ExpirationTime
from .expiration import utcnow
from .settings import AUTH_HEADERS

if TYPE_CHECKING:
    from requests import PreparedRequest

    from ..models import CachedResponse
    from . import CacheDirectives, HeaderDict
    from .settings import CacheSettings

# Final actions a request can resolve to
CACHE_HIT = 'cache_hit'  # Served from the cache without sending a request
FETCH = 'fetch'  # Sent a new request (cache miss, bypass, or refresh with new content)
REVALIDATED = 'revalidated'  # Conditional request returned 304; cached body + updated headers
STALE_WHILE_REVALIDATE = (
    'stale_while_revalidate'  # Returned stale response; refreshing in background
)
STALE_IF_ERROR = 'stale_if_error'  # Request failed; returned stale response
NOT_CACHED = 'not_cached'  # only-if-cached could not be satisfied; 504 returned

# Header names that must never appear in a trace
SENSITIVE_HEADERS = frozenset(h.lower() for h in AUTH_HEADERS)


@define(repr=False)
class KeySummary(RichMixin):
    """Summary of cache key composition. Contains no body content or header values."""

    method: str = field(default=None)
    url: str = field(default=None)
    body_sha256: Optional[str] = field(default=None)
    match_headers: List[str] = field(factory=list)
    ignored_parameters: List[str] = field(factory=list)


@define(repr=False)
class CacheCheck(RichMixin):
    """A single named node in the decision process, e.g. ``vary`` or ``only_if_cached``"""

    name: str = field(default=None)
    outcome: str = field(default=None)
    detail: Optional[str] = field(default=None)


@define(repr=False)
class ResponseGeneration(RichMixin):
    """One generation of a cached response. After a 304 revalidation there are two generations:
    the original (which still provides the body) and the revalidated one (which provides updated
    headers and expiration), linked by ``revalidated_from``.
    """

    created_at: Optional[datetime] = field(default=None)
    expires: Optional[datetime] = field(default=None)
    status_code: Optional[int] = field(default=None)
    headers: List[str] = field(factory=list)  # Non-sensitive header names only
    revalidated_from: Optional[int] = field(default=None)


@define(repr=False)
class CacheDecision(RichMixin):
    """A trace of how the cache policy arrived at its final action for a single request"""

    cache_key: Optional[str] = field(default=None)
    key: Optional[KeySummary] = field(default=None)
    now: Optional[datetime] = field(default=None)
    expire_after: ExpirationTime = field(default=None)
    expiration_source: Optional[str] = field(default=None)
    url_expiration_pattern: Optional[ExpirationPattern] = field(default=None)
    request_directives: Dict[str, Any] = field(factory=dict)
    response_directives: Dict[str, Any] = field(factory=dict)
    backend_hit: Optional[bool] = field(default=None)
    stored_created_at: Optional[datetime] = field(default=None)
    stored_expires: Optional[datetime] = field(default=None)
    stored_age: Optional[timedelta] = field(default=None)
    stored_expired: Optional[bool] = field(default=None)
    conditional_headers: Dict[str, str] = field(factory=dict)
    checks: List[CacheCheck] = field(factory=list)
    generations: List[ResponseGeneration] = field(factory=list)
    action: Optional[str] = field(default=None)

    @classmethod
    def for_request(
        cls,
        cache_key: str,
        request: 'PreparedRequest',
        settings: 'CacheSettings',
        directives: 'CacheDirectives',
        expire_after: ExpirationTime,
        read_criteria: Dict[str, bool],
        only_if_cached: bool,
        refresh: bool,
        stale_if_error: Any,
        stale_while_revalidate: Any,
    ) -> 'CacheDecision':
        """Create a decision trace from the same request info used to initialize CacheActions"""
        from .expiration import get_url_expiration_pattern

        decision = cls(
            cache_key=cache_key,
            key=summarize_key(request, settings),
            now=utcnow(),
            expire_after=expire_after,
            request_directives=directives_dict(directives),
        )

        # Expiration source, in order of precedence
        if directives.max_age is not None:
            decision.expiration_source = 'request max-age'
        elif settings.urls_expire_after:
            pattern = get_url_expiration_pattern(request.url, settings.urls_expire_after)
            if pattern is not None:
                decision.url_expiration_pattern = pattern
                decision.expiration_source = 'url pattern'
        if decision.expiration_source is None and settings.expire_after is not None:
            decision.expiration_source = 'session settings'

        # Nodes for pre-read cache checks
        decision.add_check(
            'disabled cache', 'skipped' if read_criteria['disabled cache'] else 'passed'
        )
        decision.add_check(
            'method',
            'skipped' if read_criteria['disabled method'] else 'passed',
            detail=request.method,
        )
        decision.add_check(
            'request no-cache/no-store',
            'skipped' if read_criteria['disabled by headers or refresh'] else 'passed',
        )
        decision.add_check(
            'request expiration',
            'skipped' if read_criteria['disabled by expiration'] else 'passed',
        )

        # Nodes for request features that affect the decision
        body_state = 'hashed' if decision.key and decision.key.body_sha256 else 'empty'
        decision.add_check('request body', body_state)
        decision.add_check(
            'ignored_parameters',
            'active' if settings.ignored_parameters else 'inactive',
            detail=','.join(sorted(settings.ignored_parameters or [])),
        )
        decision.add_check('only_if_cached', 'active' if only_if_cached else 'inactive')
        decision.add_check('refresh', 'active' if refresh else 'inactive')
        decision.add_check(
            'max_stale',
            'active' if directives.max_stale is not None else 'inactive',
            detail=str(directives.max_stale) if directives.max_stale is not None else None,
        )
        decision.add_check(
            'min_fresh',
            'active' if directives.min_fresh is not None else 'inactive',
            detail=str(directives.min_fresh) if directives.min_fresh is not None else None,
        )
        decision.add_check(
            'stale_if_error',
            'active' if stale_if_error else 'inactive',
            detail=str(stale_if_error) if stale_if_error else None,
        )
        decision.add_check(
            'stale_while_revalidate',
            'active' if stale_while_revalidate else 'inactive',
            detail=str(stale_while_revalidate) if stale_while_revalidate else None,
        )
        return decision

    def add_check(self, name: str, outcome: str, detail: Optional[str] = None):
        """Record a named decision node"""
        self.checks.append(CacheCheck(name=name, outcome=outcome, detail=detail))

    def get_check(self, name: str) -> Optional[CacheCheck]:
        """Get the most recent check with the given name, if any"""
        matches = [c for c in self.checks if c.name == name]
        return matches[-1] if matches else None

    def record_stored_response(self, cached_response: 'CachedResponse'):
        """Record age/expiration info and the initial generation for a stored response.
        Uses the timestamp captured at decision creation rather than re-reading the clock.
        """
        self.stored_created_at = cached_response.created_at
        self.stored_expires = cached_response.expires
        self.stored_expired = cached_response.is_expired
        if cached_response.created_at and self.now:
            self.stored_age = self.now - cached_response.created_at
        self.add_generation(
            status_code=cached_response.status_code,
            headers=cached_response.headers,
            created_at=cached_response.created_at,
            expires=cached_response.expires,
        )

    def add_generation(
        self,
        status_code: Optional[int] = None,
        headers: Optional['HeaderDict'] = None,
        created_at: Optional[datetime] = None,
        expires: Optional[datetime] = None,
        revalidated_from: Optional[int] = None,
    ) -> ResponseGeneration:
        """Append a response generation to the trace"""
        generation = ResponseGeneration(
            created_at=created_at,
            expires=expires,
            status_code=status_code,
            headers=safe_header_names(headers or {}),
            revalidated_from=revalidated_from,
        )
        self.generations.append(generation)
        return generation


def summarize_key(request: 'PreparedRequest', settings: 'CacheSettings') -> KeySummary:
    """Summarize cache key composition for a request, using the same normalization as
    :py:func:`.create_key`. The request body is recorded only as a hash, and only header names
    (never values) are recorded.
    """
    norm_request = normalize_request(
        request, settings.ignored_parameters, settings.content_root_key
    )
    body_hash = sha256(encode(norm_request.body)).hexdigest() if norm_request.body else None

    match_headers = settings.match_headers
    if match_headers is True:
        matched = sorted(
            k.lower() for k in (norm_request.headers or {}) if k.lower() not in SENSITIVE_HEADERS
        )
    elif match_headers:
        matched = sorted(
            str(k).lower() for k in match_headers if str(k).lower() not in SENSITIVE_HEADERS
        )
    else:
        matched = []

    return KeySummary(
        method=norm_request.method,
        url=norm_request.url,
        body_sha256=body_hash,
        match_headers=matched,
        ignored_parameters=sorted(str(p) for p in (settings.ignored_parameters or [])),
    )


def directives_dict(directives: 'CacheDirectives') -> Dict[str, Any]:
    """Convert cache directives to a dict of only the values that were set"""
    return {k: v for k, v in attrs.asdict(directives).items() if v not in (None, False)}


def safe_header_names(headers: 'HeaderDict') -> List[str]:
    """Get sorted header names, excluding sensitive headers"""
    return sorted(k.lower() for k in headers if k.lower() not in SENSITIVE_HEADERS)
