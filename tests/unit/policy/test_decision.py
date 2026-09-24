"""Tests for the optional CacheDecision trace (CachedSession(cache_trace=True))"""

from datetime import timedelta
from unittest.mock import patch

import pytest

from requests import Request
from requests.exceptions import RequestException

from requests_cache import ALL_METHODS, CachedSession
from requests_cache.policy.decision import (
    ACTION_ERROR_504,
    ACTION_HIT,
    ACTION_RESEND,
    ACTION_REVALIDATED,
    ACTION_SEND,
    ACTION_STALE_IF_ERROR,
)
from tests.conftest import (
    ETAG,
    MOCKED_URL,
    MOCKED_URL_200_404,
    MOCKED_URL_ETAG,
    MOCKED_URL_REDIRECT,
    MOCKED_URL_VARY,
    START_DT,
    mount_mock_adapter,
    skip_pypy,
    time_travel,
)

SECRET = 'super-secret-value'


@pytest.fixture
def traced_session(mock_session) -> CachedSession:
    """mock_session with decision tracing enabled"""
    mock_session.settings.cache_trace = True
    return mock_session


def assert_action_matches_response(decision, response):
    """The traced action must reduce to the same outcome exposed by the classic response API"""
    if decision.action == ACTION_HIT:
        assert response.from_cache is True
        assert response.is_expired is False
    elif decision.action in (ACTION_SEND, ACTION_RESEND):
        assert response.from_cache is False
    elif decision.action == ACTION_REVALIDATED:
        assert response.from_cache is True
        assert response.revalidated is True
    elif decision.action == ACTION_STALE_IF_ERROR:
        assert response.from_cache is True
        assert response.is_expired is True
    elif decision.action == ACTION_ERROR_504:
        assert response.status_code == 504
    else:
        raise AssertionError(f'Unexpected action: {decision.action}')


def test_trace_disabled_by_default(mock_session):
    """Without cache_trace, responses are unchanged and carry no decision"""
    response_1 = mock_session.get(MOCKED_URL)
    response_2 = mock_session.get(MOCKED_URL)
    assert response_1.cache_decision is None
    assert response_2.cache_decision is None
    assert response_1.from_cache is False
    assert response_2.from_cache is True


def test_trace_miss_then_fresh_hit(traced_session):
    response_1 = traced_session.get(MOCKED_URL)
    decision_1 = response_1.cache_decision
    assert decision_1.action == ACTION_SEND
    assert decision_1.backend_hit is False
    assert decision_1.cache_key == response_1.cache_key
    assert_action_matches_response(decision_1, response_1)

    response_2 = traced_session.get(MOCKED_URL)
    decision_2 = response_2.cache_decision
    assert decision_2.action == ACTION_HIT
    assert decision_2.backend_hit is True
    assert decision_2.stored_created_at is not None
    assert decision_2.cache_key == response_2.cache_key == decision_1.cache_key
    assert_action_matches_response(decision_2, response_2)


def test_trace_key_summary(traced_session):
    """The key summary describes key composition: method, URL, body, and matched headers"""
    response = traced_session.post(MOCKED_URL, data={'secretbodykey': 'secretbodyvalue'})
    summary = response.cache_decision.key_summary
    assert summary.method == 'POST'
    assert summary.url.endswith('requests-cache.com/text')
    assert summary.body_in_key is True
    assert summary.body_size > 0
    assert summary.matched_headers == []
    assert summary.serializer == 'pickle'
    # Body content must not appear anywhere in the trace
    assert 'secretbodykey' not in repr(response.cache_decision)
    assert 'secretbodyvalue' not in repr(response.cache_decision)


def test_trace_key_summary__match_headers(traced_session):
    """Matched headers are recorded by name only, never by value"""
    traced_session.settings.match_headers = ['Accept']
    response = traced_session.get(MOCKED_URL, headers={'Accept': 'application/json'})
    summary = response.cache_decision.key_summary
    assert summary.matched_headers == ['accept']
    assert 'application/json' not in repr(response.cache_decision)


def test_trace_sensitive_data_redacted(traced_session):
    """Sensitive headers (e.g. Authorization) must not appear in the trace"""
    response = traced_session.get(MOCKED_URL, headers={'Authorization': f'Bearer {SECRET}'})
    decision = response.cache_decision
    assert SECRET not in repr(decision)
    assert 'authorization' not in decision.key_summary.matched_headers
    assert 'Authorization' in decision.key_summary.ignored_parameters


@skip_pypy
def test_trace_expired_resend(traced_session):
    """An expired cached response leads to a resend, with stored/now ages recorded"""
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(MOCKED_URL)

        with time_travel(START_DT + timedelta(seconds=1.1)):
            response = traced_session.get(MOCKED_URL)

            decision = response.cache_decision
            assert decision.action == ACTION_RESEND
            assert decision.backend_hit is True
            assert decision.now is not None
            assert decision.now >= START_DT + timedelta(seconds=1)
            assert decision.stored_expires is not None
            assert decision.stored_expires < decision.now
            assert decision.stored_age is not None
            assert decision.stored_age >= timedelta(seconds=1)
            assert_action_matches_response(decision, response)


def test_trace_304_revalidation(traced_session):
    """A 304 revalidation links two generations: old body + new headers"""
    url = f'{MOCKED_URL_ETAG}/traced'
    traced_session.mock_adapter.register_uri(
        'GET', url, headers={'ETag': ETAG}, text='original content'
    )
    traced_session.get(url)

    traced_session.mock_adapter.register_uri(
        'GET',
        url,
        status_code=304,
        headers={'ETag': ETAG, 'X-New-Header': 'new', 'X-API-Key': SECRET},
    )
    response = traced_session.get(url, refresh=True)
    decision = response.cache_decision

    assert decision.action == ACTION_REVALIDATED
    assert decision.refresh is True
    assert decision.conditional_headers == {'If-None-Match': ETAG}
    assert_action_matches_response(decision, response)

    # The response links the old generation (body) with the new generation (headers)
    assert len(decision.generations) == 2
    stored, origin = decision.generations
    assert stored.source == 'stored'
    assert stored.status_code == 200
    assert stored.body_size == len('original content')
    assert stored.created_at == response.created_at
    assert origin.source == 'origin'
    assert origin.status_code == 304
    assert origin.body_size == 0
    assert origin.headers['X-New-Header'] == 'new'
    assert response.headers['X-New-Header'] == 'new'
    assert response.text == 'original content'
    # Sensitive headers from the 304 response must not enter the trace
    assert 'X-API-Key' not in origin.headers
    assert SECRET not in repr(decision)


@skip_pypy
def test_trace_stale_if_error(traced_session):
    """With stale_if_error, a server error results in returning the stale cached response"""
    traced_session.settings.stale_if_error = True
    traced_session.settings.expire_after = 1
    traced_session.settings.allowable_codes = (200,)

    with time_travel(START_DT):
        assert traced_session.get(MOCKED_URL_200_404).status_code == 200

        with time_travel(START_DT + timedelta(seconds=1.1)):
            response = traced_session.get(MOCKED_URL_200_404)

            decision = response.cache_decision
            assert decision.action == ACTION_STALE_IF_ERROR
            assert decision.stale_if_error is True
            assert decision.backend_hit is True
            assert_action_matches_response(decision, response)


@skip_pypy
def test_trace_stale_if_error__exception(traced_session):
    """stale_if_error also applies when the request raises an exception"""
    traced_session.settings.stale_if_error = True
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(MOCKED_URL)

        with patch.object(traced_session.cache, 'save_response', side_effect=RequestException):
            with time_travel(START_DT + timedelta(seconds=1.1)):
                response = traced_session.get(MOCKED_URL)

                assert response.cache_decision.action == ACTION_STALE_IF_ERROR
                assert_action_matches_response(response.cache_decision, response)


def test_trace_redirect(traced_session):
    response_1 = traced_session.get(MOCKED_URL_REDIRECT)
    assert response_1.cache_decision.action == ACTION_SEND
    assert_action_matches_response(response_1.cache_decision, response_1)

    response_2 = traced_session.get(MOCKED_URL_REDIRECT)
    assert response_2.cache_decision.action == ACTION_HIT
    assert_action_matches_response(response_2.cache_decision, response_2)


def test_trace_vary_mismatch(traced_session):
    """A Vary mismatch is recorded, along with the secondary Vary-qualified key"""
    headers_1 = {'Accept': 'application/json'}
    headers_2 = {'Accept': 'text/plain'}
    traced_session.get(MOCKED_URL_VARY, headers=headers_1)

    # Mismatched Vary header: primary entry doesn't match; secondary lookup misses
    response_1 = traced_session.get(MOCKED_URL_VARY, headers=headers_2)
    decision = response_1.cache_decision
    assert decision.vary.header == 'Accept'
    assert decision.vary.matched is False
    assert decision.vary.vary_cache_key is not None
    assert decision.vary.secondary_hit is False
    assert decision.cache_key == decision.vary.vary_cache_key
    assert decision.action == ACTION_SEND
    assert_action_matches_response(decision, response_1)

    # Same headers again: secondary Vary-qualified lookup hits
    response_2 = traced_session.get(MOCKED_URL_VARY, headers=headers_2)
    decision = response_2.cache_decision
    assert decision.vary.matched is False  # Primary entry still doesn't match
    assert decision.vary.secondary_hit is True
    assert decision.action == ACTION_HIT
    assert_action_matches_response(decision, response_2)


def test_trace_only_if_cached__uncached(traced_session):
    response = traced_session.get(MOCKED_URL, only_if_cached=True)
    decision = response.cache_decision
    assert decision.action == ACTION_ERROR_504
    assert decision.only_if_cached is True
    assert decision.backend_hit is False
    assert_action_matches_response(decision, response)


def test_trace_only_if_cached__cached(traced_session):
    traced_session.get(MOCKED_URL)
    response = traced_session.get(MOCKED_URL, only_if_cached=True)
    decision = response.cache_decision
    assert decision.action == ACTION_HIT
    assert decision.only_if_cached is True
    assert_action_matches_response(decision, response)


@skip_pypy
def test_trace_max_stale(traced_session):
    """max-stale allows using an expired response, and is recorded as a request directive"""
    traced_session.settings.expire_after = 1
    with time_travel(START_DT):
        traced_session.get(MOCKED_URL)

        with time_travel(START_DT + timedelta(seconds=1.1)):
            response = traced_session.get(MOCKED_URL, headers={'Cache-Control': 'max-stale=60'})

            decision = response.cache_decision
            assert decision.action == ACTION_HIT
            assert decision.max_stale == 60
            assert decision.request_directives.max_stale == 60
            assert response.from_cache is True
            assert response.is_expired is True


def test_trace_no_cache_request_directive(traced_session):
    """no-cache skips the cache read; the reason is recorded"""
    traced_session.get(MOCKED_URL)
    response = traced_session.get(MOCKED_URL, headers={'Cache-Control': 'no-cache'})
    decision = response.cache_decision
    assert decision.action == ACTION_SEND
    assert decision.request_directives.no_cache is True
    assert 'disabled by headers or refresh' in decision.read_skip_reasons
    assert_action_matches_response(decision, response)


def test_trace_disabled_cache(traced_session):
    """A disabled cache has an explicit node and skip reason"""
    with traced_session.cache_disabled():
        response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert decision.disabled is True
    assert 'disabled cache' in decision.read_skip_reasons
    assert decision.action == ACTION_SEND
    assert_action_matches_response(decision, response)


def test_trace_url_rule_expiration_source(traced_session):
    """A matching per-URL expiration pattern is recorded as the expiration source"""
    traced_session.settings.urls_expire_after = {'requests-cache.com/text': 30}
    response = traced_session.get(MOCKED_URL)
    decision = response.cache_decision
    assert decision.url_rule == 'requests-cache.com/text'
    assert decision.expiration_source == 'url_rule'
    assert decision.expire_after == 30


def test_trace_request_directive_expiration_source(traced_session):
    """A request Cache-Control max-age takes precedence and is recorded as the source"""
    traced_session.settings.urls_expire_after = {'requests-cache.com/text': 30}
    response = traced_session.get(MOCKED_URL, headers={'Cache-Control': 'max-age=60'})
    decision = response.cache_decision
    assert decision.expiration_source == 'request'
    assert decision.expire_after == 60
    assert decision.request_directives.max_age == 60


def test_trace_response_directive_expiration_source(traced_session):
    """With cache_control enabled, response Cache-Control headers are the expiration source"""
    url = f'{MOCKED_URL}/cache-control'
    traced_session.settings.cache_control = True
    traced_session.mock_adapter.register_uri(
        'GET', url, headers={'Cache-Control': 'max-age=120'}, text='mock response'
    )
    response = traced_session.get(url)
    decision = response.cache_decision
    assert decision.expiration_source == 'response'
    assert decision.expire_after == 120
    assert decision.response_directives.max_age == 120


def test_trace_session_expiration_source(traced_session):
    response = traced_session.get(MOCKED_URL)
    assert response.cache_decision.expiration_source == 'session'


def test_trace_not_serialized(traced_session, tempfile_path):
    """The decision trace is never persisted; deserialized responses rebuild it per request"""
    traced_session.get(MOCKED_URL)
    traced_session.close()

    # Reopen the same cache with a new session, with tracing enabled
    new_session = CachedSession(
        cache_name=tempfile_path,
        backend='sqlite',
        allowable_methods=ALL_METHODS,
        cache_trace=True,
    )
    request = Request('GET', MOCKED_URL).prepare()
    cached = new_session.cache.get_response(new_session.cache.create_key(request))
    assert cached is not None
    assert cached.cache_decision is None

    response = new_session.get(MOCKED_URL)
    assert response.from_cache is True
    assert response.cache_decision is not None
    assert response.cache_decision.action == ACTION_HIT
    new_session.close()


def test_trace_consistent_with_untraced_session(traced_session, tmp_path):
    """Enabling the trace must not change cache behavior or classic response attributes"""
    plain_session = mount_mock_adapter(
        CachedSession(
            cache_name=str(tmp_path / 'plain.db'),
            backend='sqlite',
            allowable_methods=ALL_METHODS,
        )
    )
    urls = [MOCKED_URL, MOCKED_URL, MOCKED_URL_ETAG, MOCKED_URL_ETAG]
    for url in urls:
        traced_response = traced_session.get(url)
        plain_response = plain_session.get(url)
        assert traced_response.from_cache == plain_response.from_cache
        assert traced_response.status_code == plain_response.status_code
        assert traced_response.cache_key == plain_response.cache_key
        assert traced_response.text == plain_response.text
        assert traced_response.cache_decision is not None
        assert plain_response.cache_decision is None
    plain_session.close()
