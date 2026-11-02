"""Phase 2: Model C in shadow mode through the FastAPI app. Small world, real packaged artifact."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval import phase2_shadow as p2
from eval.phase3 import _same
from eval.sequences_v4 import assemble_v4, generate_world, rebuild_features_at
from services.app import main, shadow
from services.app.adapters import diagnosis_v4
from services.app.domain.deployment_features import DEPLOYMENT_CORE
from services.app.domain.taxonomy import Cause

pytestmark = pytest.mark.skipif(not (Path('eval/artifacts/phase4') / 'model.joblib').exists(), reason='Phase 4 artifact not present')


@pytest.fixture(scope='module')
def world():
    return generate_world(500, 91001, scenario='typical')


@pytest.fixture(scope='module')
def replay(world):
    """Replay once through the app; capture the exact feature dict handed to the model for every scored failure."""
    captured, original = [], diagnosis_v4.diagnose

    def spy(**kw):
        captured.append(kw['features'])
        return original(**kw)

    shadow.reset()
    diagnosis_v4.diagnose = spy
    try:
        with TestClient(main.app) as client:
            truth, responses = p2.replay_through_api(world, client)
    finally:
        diagnosis_v4.diagnose = original
    return truth, responses, dict(zip(truth, captured))      # scored failures arrive in `truth` order


def test_model_input_has_all_43_features_and_no_label(replay):
    truth, responses, captured = replay
    assert len(responses) > 20 and all(r['status'] == 'scored' for r in responses.values())
    art = diagnosis_v4.bundle()
    assert list(art['model'].features) == list(DEPLOYMENT_CORE) and len(DEPLOYMENT_CORE) == 43
    assert len(captured) == len(truth)
    for features in captured.values():
        assert set(DEPLOYMENT_CORE) <= set(features)
        assert not {'true_cause', 'n_causes_fired', 'actual_cause'} & set(features)


def test_features_equal_a_replay_that_physically_excludes_later_events(world, replay):
    truth, _, captured = replay
    for pid in list(truth)[::max(1, len(truth) // 4)][:4]:
        rebuilt = rebuild_features_at(world, pid)
        assert all(_same(captured[pid][k], v) for k, v in rebuilt.items()), pid


def test_schema_rejects_a_label_and_the_label_never_changes_the_prediction(world):
    client = TestClient(main.app)
    row = next(r for r in world.rows if r['status'] == 'failed' and not r['is_warmup'])
    body = p2.event_payload(row, diagnose=True, observe=False)
    assert client.post('/shadow/event', json={**body, 'true_cause': 'CARD_EXPIRED'}).status_code == 422
    shadow.reset()
    a = client.post('/shadow/event', json=body).json()
    shadow.attach_label(row['payment_id'], 'CARD_EXPIRED')                       # evaluation-only side channel
    b = client.get('/shadow/diagnoses', params={'payment_id': row['payment_id']}).json()
    for key in ('predicted_cause', 'confidence', 'probabilities', 'policy_cause'):
        assert a[key] == b[key]
    assert a['actual_cause'] is None and b['actual_cause'] == 'CARD_EXPIRED'


def test_prediction_and_confidence_are_returned_and_consistent(replay):
    _, responses, _ = replay
    for r in responses.values():
        assert r['confidence'] == pytest.approx(max(r['probabilities'].values()))
        assert sum(r['probabilities'].values()) == pytest.approx(1)
        assert r['predicted_cause'] == max(r['probabilities'], key=r['probabilities'].get)
        assert r['low_confidence'] == (r['confidence'] < 0.60) and r['model_id'].startswith('phase4-hgb')


def test_out_of_order_and_missing_identifiers_fail_safely():
    shadow.reset()
    t = datetime(2026, 5, 1, 12, tzinfo=timezone.utc)
    base = dict(method='card', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='issuer_bank', gw_error_step='payment_authorization')
    first = shadow.process(shadow.make_observation('p1', occurred_at=t, amount_minor=49900, **base))   # no customer / issuer / instrument
    late = shadow.process(shadow.make_observation('p0', occurred_at=t - timedelta(hours=1), amount_minor=49900, **base))
    bad = shadow.process(shadow.make_observation('p2', occurred_at=t, **base))                  # amount missing -> rejected, not a crash
    assert bad['status'] == 'invalid_event' and bad['predicted_cause'] is None
    assert first['status'] == 'scored' and late['status'] == 'out_of_order' and late['predicted_cause'] is None
    # an unknown customer does not share one global history bucket
    assert shadow.make_observation('a', occurred_at=t).customer_id != shadow.make_observation('b', occurred_at=t).customer_id
