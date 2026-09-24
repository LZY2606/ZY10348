"""Tests for the optional CacheDecision trace (CacheSettings.trace_decisions)"""

from datetime import timedelta
from hashlib import sha256
from unittest.mock import patch

import pytest
from requests import Request

from requests_cache import CachedSession
from requests_cache._utils import encode
from requests_cache.cache_keys import normalize_request
from requests_cache.policy import (
    CACHE_HIT,
    FETCH,
    NOT_CACHED,
    REVALIDATED,
    STALE_IF_ERROR,
    CacheDecision,
)
from tests.conftest import (
    ETAG,
    MOCKED_URL,
    MOCKED_URL_JSON,
    MOCKED_URL_REDIRECT,
    MOCKED_URL_VARY,
    START_DT,
    skip_pypy,
    time_travel,
)


@pytest.fixture(scope='function')
def traced_session(mock_session) -> CachedSession:
    """A mock_session with decision tracing enabled"""
    mock_session.settings.trace_decisions = True
    yield mock_session


def assert_action_matches_response(response):
    """The traced action must be reducible to the response's existing (old API) attributes"""
    decision = response.cache_decision
    assert isinstance(decision, CacheDecision)
    if decision.action == CACHE_HIT:
        assert response.from_cache is True
        assert response.revalidated is False
    elif decision.action == FETCH:
        assert response.from_cache is False
    elif decision.action == REVALIDATED:
        assert response.from_cache is True
        assert response.revalidated is True
    elif decision.action == STALE_IF_ERROR:
        assert response.from_cache is True
        assert response.is_expired is True
    elif decision.action == NOT_CACHED:
        assert response.status_code == 504


def test_trace_disabled_by_default(mock_session):
    """By default, no trace is created and the response object is unchanged"""
    response = mock_session.get(MOCKED_URL)
    assert response.cache_decision is None
    response = mock_session.get(MOCKED_URL)
    assert response.cache_decision is None


def test_fresh_cache_hit(traced_session):
    traced_session.settings.expire_after = 60
    response_1 = traced_session.get(MOCKED_URL)
    decision_1 = response_1.cache_decision
    assert_action_matches_response(response_1)
    assert decision_1.action == FETCH
    assert decision_1.backend_hit is False
    assert decision_1.expiration_source == 'session settings'
    assert decision_1.get_check('freshness').outcome == 'miss'
    assert decision_1.get_check('cache write').outcome == 'allowed'
    # The newly stored response is recorded as a generation
    assert len(decision_1.generations) == 1
    assert decision_1.generations[0].status_code == 200

    response_2 = traced_session.get(MOCKED_URL)
    decision_2 = response_2.cache_decision
    assert_action_matches_response(response_2)
    assert decision_2.action == CACHE_HIT
    assert decision_2.backend_hit is True
    assert decision_2.stored_expired is False
    assert decision_2.get_check('freshness').outcome == 'fresh'
    # The trace observes stored values; age is derived from the single recorded timestamp
    assert decision_2.stored_age == decision_2.now - decision_2.stored_created_at
    assert response_2.cache_key == decision_2.cache_key


@skip_pypy
def test_expired_refetch(traced_session):
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(MOCKED_URL)

    with time_travel(START_DT + timedelta(seconds=2)):
        response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == FETCH
    assert decision.backend_hit is True
    assert decision.stored_expired is True
    assert decision.get_check('freshness').outcome == 'stale'
    # Stored generation + refetched generation
    assert len(decision.generations) == 2
    assert decision.generations[1].revalidated_from is None


@skip_pypy
def test_revalidated_304(traced_session):
    """A 304 revalidation links the old body (generation 0) with new headers (generation 1)"""
    url = f'{MOCKED_URL}/revalidate'
    traced_session.mock_adapter.register_uri(
        'GET',
        url,
        [
            {'status_code': 200, 'headers': {'ETag': ETAG}, 'text': 'original body'},
            {'status_code': 304, 'headers': {'ETag': ETAG, 'X-New-Header': 'new'}},
        ],
    )
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(url)

    with time_travel(START_DT + timedelta(seconds=2)):
        response = traced_session.get(url)

    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == REVALIDATED
    assert decision.conditional_headers == {'If-None-Match': ETAG}
    assert decision.stored_expired is True

    # Two linked generations: stored response (body source) and revalidated headers
    assert len(decision.generations) == 2
    stored, revalidated = decision.generations
    assert revalidated.revalidated_from == 0
    assert 'x-new-header' not in stored.headers
    assert 'x-new-header' in revalidated.headers

    # Old API results are consistent: old body is kept, new headers are merged
    assert response.text == 'original body'
    assert response.headers['X-New-Header'] == 'new'
    assert response.from_cache is True
    assert response.revalidated is True


@skip_pypy
def test_stale_if_error__server_error(traced_session):
    url = f'{MOCKED_URL}/flaky'
    traced_session.mock_adapter.register_uri(
        'GET', url, [{'status_code': 200, 'text': 'ok'}, {'status_code': 500}]
    )
    traced_session.settings.stale_if_error = True
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        assert traced_session.get(url).status_code == 200

    with time_travel(START_DT + timedelta(seconds=2)):
        response = traced_session.get(url)
        decision = response.cache_decision
        assert_action_matches_response(response)
        assert decision.action == STALE_IF_ERROR
        assert decision.get_check('stale_if_error').outcome == 'used stale response'
        assert response.status_code == 200
        assert response.text == 'ok'


def test_redirect(traced_session):
    response_1 = traced_session.get(MOCKED_URL_REDIRECT)
    assert_action_matches_response(response_1)
    assert response_1.cache_decision.action == FETCH
    assert response_1.status_code == 200

    # The redirect target is cached, so the second request is served from the cache
    response_2 = traced_session.get(MOCKED_URL_REDIRECT)
    decision = response_2.cache_decision
    assert_action_matches_response(response_2)
    assert decision.action == CACHE_HIT
    assert decision.backend_hit is True


def test_vary_mismatch(traced_session):
    headers_1 = {'Accept': 'text/plain'}
    headers_2 = {'Accept': 'application/json'}

    response_1 = traced_session.get(MOCKED_URL_VARY, headers=headers_1)
    assert response_1.cache_decision.action == FETCH

    # A cached response with a matching Vary header is a cache hit
    response_2 = traced_session.get(MOCKED_URL_VARY, headers=headers_1)
    decision_2 = response_2.cache_decision
    assert_action_matches_response(response_2)
    assert decision_2.action == CACHE_HIT
    assert decision_2.get_check('vary').outcome == 'matched'
    assert decision_2.get_check('vary').detail == 'Accept'

    # A mismatched Vary header is a cache miss, retried with a Vary-qualified key
    response_3 = traced_session.get(MOCKED_URL_VARY, headers=headers_2)
    decision_3 = response_3.cache_decision
    assert_action_matches_response(response_3)
    assert decision_3.action == FETCH
    assert decision_3.get_check('vary').outcome == 'mismatch'
    assert decision_3.get_check('vary secondary key').outcome == 'miss'

    # The Vary-qualified entry is now cached as well
    response_4 = traced_session.get(MOCKED_URL_VARY, headers=headers_2)
    decision_4 = response_4.cache_decision
    assert_action_matches_response(response_4)
    assert decision_4.action == CACHE_HIT
    assert decision_4.get_check('vary secondary key').outcome == 'hit'


def test_only_if_cached__miss(traced_session):
    response = traced_session.get(MOCKED_URL, only_if_cached=True)
    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == NOT_CACHED
    assert decision.backend_hit is False
    assert decision.get_check('only_if_cached').outcome == 'unsatisfiable'
    assert decision.request_directives.get('only_if_cached') is True


@skip_pypy
def test_max_stale(traced_session):
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(MOCKED_URL)

    with time_travel(START_DT + timedelta(seconds=2)):
        response = traced_session.get(MOCKED_URL, headers={'Cache-Control': 'max-stale=60'})

    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == CACHE_HIT
    assert decision.stored_expired is True
    assert decision.get_check('max_stale').outcome == 'active'
    assert decision.get_check('max_stale').detail == '60'
    assert decision.request_directives.get('max_stale') == 60


def test_request_directives__no_cache(traced_session):
    traced_session.get(MOCKED_URL)
    response = traced_session.get(MOCKED_URL, headers={'Cache-Control': 'no-cache'})
    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == FETCH
    # The cache was never read, so there is no backend hit to record
    assert decision.backend_hit is None
    assert decision.get_check('request no-cache/no-store').outcome == 'skipped'
    assert decision.request_directives.get('no_cache') is True


def test_disabled_cache(traced_session):
    with traced_session.cache_disabled():
        response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert_action_matches_response(response)
    assert decision.action == FETCH
    assert decision.backend_hit is None
    assert decision.get_check('disabled cache').outcome == 'skipped'
    assert decision.get_check('cache write').outcome == 'skipped'


def test_url_expiration_pattern(traced_session):
    traced_session.settings.urls_expire_after = {'requests-cache.com/text': 30}
    response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert decision.url_expiration_pattern == 'requests-cache.com/text'
    assert decision.expiration_source == 'url pattern'
    assert decision.expire_after == 30


def test_response_directives(traced_session):
    url = f'{MOCKED_URL}/cache-control'
    traced_session.mock_adapter.register_uri(
        'GET', url, headers={'Cache-Control': 'max-age=120'}, text='mock response'
    )
    traced_session.settings.cache_control = True
    response = traced_session.get(url)
    decision = response.cache_decision
    assert decision.response_directives == {'max_age': 120}
    assert decision.expiration_source == 'response max-age'
    assert decision.expire_after == 120


def test_key_summary__method_body_and_sensitive_data(traced_session):
    """The key summary records method, normalized URL, and a body hash; sensitive header values,
    ignored parameters, and body content must never appear in the trace.
    """
    body = {'api_key': 'secret-key-123', 'data': 'payload'}
    headers = {'Authorization': 'Bearer secret-token-123', 'X-Custom': 'value'}
    traced_session.settings.match_headers = True
    response = traced_session.post(
        MOCKED_URL_JSON, json=body, params={'api_key': 'secret-key-123'}, headers=headers
    )
    decision = response.cache_decision
    assert_action_matches_response(response)

    # Method and body are explicit nodes in the trace
    assert decision.key.method == 'POST'
    assert decision.get_check('method').detail == 'POST'
    assert decision.get_check('request body').outcome == 'hashed'
    assert decision.get_check('ignored_parameters').outcome == 'active'

    # The body hash matches the normalized body (with ignored parameters removed)
    request = Request('POST', MOCKED_URL_JSON, json=body).prepare()
    norm_request = normalize_request(request, traced_session.settings.ignored_parameters)
    assert decision.key.body_sha256 == sha256(encode(norm_request.body)).hexdigest()

    # Matched header names are recorded, but not sensitive ones and never values
    assert 'x-custom' in decision.key.match_headers
    assert 'authorization' not in decision.key.match_headers

    # No sensitive data anywhere in the trace
    trace_str = str(decision)
    assert 'secret-key-123' not in trace_str
    assert 'secret-token-123' not in trace_str
    assert 'payload' not in trace_str
    # Ignored URL parameters are redacted in the same way as in the actual cache key
    assert 'api_key=REDACTED' in decision.key.url


def test_trace_observes_single_backend_read(traced_session):
    """The trace records the backend hit from the request's own cache read; it must not
    perform any additional backend reads of its own.
    """
    traced_session.get(MOCKED_URL)
    with patch.object(
        traced_session.cache, 'get_response', wraps=traced_session.cache.get_response
    ) as spy:
        response = traced_session.get(MOCKED_URL)
        assert spy.call_count == 1
        assert response.cache_decision.backend_hit is True


@skip_pypy
def test_trace_observes_single_timestamp(traced_session):
    """The trace captures one timestamp per request instead of re-reading the clock"""
    with time_travel(START_DT):
        response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert decision.now is not None
    assert abs((decision.now - START_DT).total_seconds()) <= 2
    assert decision.generations[0].created_at == decision.now
