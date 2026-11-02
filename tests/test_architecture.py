"""The product runs ONE workflow, RevenueRecoveryWorkflow, with ONE diagnosis per event. These checks prove the superseded stack is gone:
no import, registration, route, setting or infrastructure for it remains anywhere in the repository."""
import ast
import importlib
import re
from pathlib import Path

import pytest

from services.app import main
from services.app.settings import Settings

SERVICES = Path('services')
LEGACY_NAMES = ('RecoveryWorkflow', 'AdaptiveRecoveryWorkflow', 'EvaluatedRecoveryWorkflow', 'prepare_cycle', 'prepare_evaluated_cycle', 'next_hypothesis', 'reassess',
                'workflow_dispatch', 'dispatch_loop', 'evaluate_cycle', 'register_first_cycle')


def py_files():
    return [p for p in SERVICES.rglob('*.py') if '__pycache__' not in p.parts]


def test_nothing_in_the_repository_imports_the_removed_legacy_stack():
    for path in [*py_files(), *(p for d in ('eval', 'tests') for p in Path(d).rglob('*.py') if p.name != 'test_architecture.py')]:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert name != 'eval.legacy' and not name.startswith('eval.legacy.'), f'{path} imports {name}'
                assert not name.startswith(('services.app.dispatch', 'services.app.workflows.recovery', 'services.app.workflows.cycles', 'services.app.workflows.evaluated')), (path, name)


def test_the_legacy_stack_is_gone_from_the_product_tree():
    for gone in ('workflows/recovery.py', 'workflows/cycles.py', 'workflows/evaluated.py', 'workflows/activities.py', 'dispatch.py', 'test_runs.py', 'test_progress.py',
                 'domain/reassess.py', 'domain/evaluator.py', 'domain/plan_validation.py', 'domain/policy.py', 'domain/settlements.py', 'adapters/channels.py',
                 'adapters/llm_triage.py', 'adapters/razorpay_client.py'):
        assert not (SERVICES / 'app' / gone).exists(), gone
    assert sorted(p.name for p in (SERVICES / 'app' / 'workflows').glob('*.py')) == ['__init__.py', 'revenue.py', 'revenue_activities.py']


def test_no_legacy_workflow_diagnosis_or_reassessment_name_survives_in_services():
    for path in py_files():
        text = path.read_text(encoding='utf-8')
        for name in LEGACY_NAMES:
            assert not re.search(r'' + name + r'', text), f'{path} still mentions {name}'


def test_the_worker_registers_only_the_revenue_workflow_and_its_activities():
    from services.app.workflows import revenue, revenue_activities
    worker_src = Path('services/app/worker.py').read_text(encoding='utf-8')
    build_src = Path('services/app/revenue/worker.py').read_text(encoding='utf-8')
    assert 'workflows=[RevenueRecoveryWorkflow]' in build_src and 'activities=revenue_activities.ACTIVITIES' in build_src
    assert 'build(client).run()' in worker_src and 'Workflow' not in worker_src.replace('RevenueRecoveryWorkflow', '')
    assert [a.__name__ for a in revenue_activities.ACTIVITIES] == ['rv_start', 'rv_execute_step', 'rv_simulate_response', 'rv_settle_request', 'rv_finish']
    assert revenue.RevenueRecoveryWorkflow.__name__ == 'RevenueRecoveryWorkflow'


def test_only_the_current_routes_are_served():
    paths = {r.path for r in main.app.routes}
    for gone in ('/webhooks/razorpay', '/simulate/failure', '/simulate/model-c', '/simulate/model-c-evaluated', '/simulate/truth/{payment_id}', '/diagnose', '/tests/payments',
                 '/tests/payments/custom', '/tests/payments/{test_id}/progress', '/tests/runs', '/tests/runs/{run_id}', '/payments/{payment_id}/recovery',
                 '/payments/{payment_id}/timeline', '/metrics/uplift', '/metrics/by-cause', '/interventions/temporal/{payment_id}'):
        assert gone not in paths, gone
    for present in ('/revenue/events', '/revenue/events/from-dataset', '/revenue/requests/{request_id}/respond', '/revenue/runs', '/pay/{request_id}', '/interventions',
                    '/interventions/revenue/{event_id}', '/tests/custom-options', '/tests/datasets', '/tests/csv/validate', '/ui', '/health', '/audit/verify', '/shadow/event'):
        assert present in paths, present
    assert not [r for r in main.app.routes if getattr(r, 'deprecated', False)]


def test_settings_hold_no_dead_configuration():
    assert set(Settings.model_fields) == {'database_url', 'temporal_address', 'demo_mode', 'ui_data_dir', 'revenue_task_queue', 'revenue_response_timeout_s', 'revenue_time_scale'}


def test_the_quarantine_and_the_in_repo_archive_are_gone_and_the_remaining_eval_support_imports():
    assert not Path('eval/legacy').exists() and not Path('archive').exists()
    for module in ('eval.build_ui_datasets', 'eval.oracle_plan', 'eval.oracle_plan_validation', 'eval.phase3_heldout', 'eval.classifier', 'eval.phase2_shadow', 'eval.legacy_generate'):
        importlib.import_module(module)
    classifier = Path('eval/classifier.py').read_text(encoding='utf-8')
    assert not any(w in classifier for w in ('llm_triage', 'use_llm', '_tier3', 'api_key'))                 # no LLM tier in the classifier


def test_the_compose_stack_holds_only_what_the_architecture_needs():
    text = Path('docker-compose.yml').read_text(encoding='utf-8')
    body = re.search(r'^services:\n(.*?)^volumes:', text, re.M | re.S).group(1)
    services = re.findall(r'^  ([a-z-]+):\s*$', body, re.M)
    assert sorted(services) == ['api', 'postgres', 'temporal', 'temporal-ui', 'worker'], services      # no redis: nothing reads it
    assert 'redis' not in text.lower() and 'REDIS' not in Path('.env.example').read_text(encoding='utf-8')
    assert 'redis' not in Path('services/app/requirements.txt').read_text(encoding='utf-8').lower()


def test_every_setting_the_product_reads_exists():
    """A renamed or removed setting must not leave a caller behind (it would only fail when a workflow is started)."""
    import re
    from services.app.settings import settings
    for path in py_files():
        for name in set(re.findall(r'settings\.([a-z_]+)', path.read_text(encoding='utf-8'))):
            assert hasattr(settings, name), f'{path} reads settings.{name}, which does not exist'
    from services.app.revenue import service
    opts = service.workflow_options('manual')
    assert opts['time_scale'] == settings.revenue_time_scale and opts['response_timeout_s'] == settings.revenue_response_timeout_s and opts['responder'] == 'manual'


def test_no_make_target_wipes_volumes_and_matplotlib_stays_out_of_production_requirements():
    make = Path('Makefile').read_text(encoding='utf-8')
    assert 'down -v' not in make and not re.search(r'^reset:', make, re.M)
    assert 'matplotlib' not in Path('services/app/requirements.txt').read_text(encoding='utf-8').lower()
    assert 'matplotlib' in Path('services/app/requirements-test.txt').read_text(encoding='utf-8')
