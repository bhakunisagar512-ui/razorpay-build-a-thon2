"""RevenueRecoveryWorkflow: Temporal runs a recovery plan that was diagnosed exactly once before it started.

For each planned step: wait the (scaled) delay, apply the stopping rules, execute, create a simulated payment request when the step collects
money, then WAIT for ACCEPT or REJECT. ACCEPT means the simulated customer completed the payment: the workflow stops. REJECT, or no answer
before the timeout, moves on to the next permitted step. When the plan is used up the event ends SIMULATED_FAILED or goes to a human.

The workflow never diagnoses, never calls Model C and never reads a hidden label. The database is the source of truth for every request: a
signal only wakes the workflow, so a duplicate or late response cannot resolve a request twice or count a recovery twice.
"""
import asyncio
from datetime import datetime, timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from services.app.workflows import revenue_activities as A

MAX_REAL_PAUSE_S = 3        # a simulated delay never costs more than this in real time
MAX_WAIT_LOOPS = 6


@workflow.defn
class RevenueRecoveryWorkflow:
    def __init__(self) -> None:
        self._responses: dict = {}

    @workflow.signal
    def payment_response(self, request_id: str, decision: str) -> None:
        self._responses.setdefault(request_id, decision)           # duplicates are ignored; the database decides

    @workflow.run
    async def run(self, event_id: str, opts: dict) -> str:
        t, retry = timedelta(seconds=30), RetryPolicy(maximum_attempts=5)
        scale = max(int(opts.get('time_scale', 100000)), 1)
        timeout_s = int(opts.get('response_timeout_s', 900))
        simulator = opts.get('responder') == 'simulator'

        def pause(seconds: float) -> int:
            return min(int(seconds // scale), MAX_REAL_PAUSE_S)

        started = await workflow.execute_activity(A.rv_start, args=[event_id, workflow.info().workflow_id], start_to_close_timeout=t, retry_policy=retry)
        sim_now = datetime.fromisoformat(started['sim_now'])
        recovered = False
        for step in started['plan']['steps']:
            if started['status'] != 'PENDING':
                break
            sim_now += timedelta(seconds=step['delay_s'])
            if pause(step['delay_s']):
                await asyncio.sleep(pause(step['delay_s']))
            result = {'outcome': 'closed'}
            for _ in range(MAX_WAIT_LOOPS):
                result = await workflow.execute_activity(A.rv_execute_step, args=[event_id, step['idx'], sim_now.isoformat(), timeout_s], start_to_close_timeout=t, retry_policy=retry)
                if result['outcome'] != 'waiting':
                    break
                resume = datetime.fromisoformat(result['resume_at'])    # a temporary rule (quiet hours, weekly cap): wait it out on the simulated clock
                gap = (resume - sim_now).total_seconds()
                sim_now = resume
                if pause(gap):
                    await asyncio.sleep(pause(gap))
            if result['outcome'] in ('closed', 'escalated', 'resolved_paid'):
                recovered = recovered or result['outcome'] == 'resolved_paid'
                break
            request_id = result.get('request_id')
            if result['outcome'] != 'executed' or not request_id:
                continue
            if simulator:
                decision = await workflow.execute_activity(A.rv_simulate_response, args=[event_id, request_id], start_to_close_timeout=t, retry_policy=retry)
            else:
                try:
                    await workflow.wait_condition(lambda rid=request_id: rid in self._responses, timeout=timedelta(seconds=timeout_s))
                except asyncio.TimeoutError:
                    pass
                decision = await workflow.execute_activity(A.rv_settle_request, args=[request_id], start_to_close_timeout=t, retry_policy=retry)
            if decision == 'ACCEPTED':
                recovered = True
                break
        return await workflow.execute_activity(A.rv_finish, args=[event_id, 'accepted' if recovered else 'exhausted'], start_to_close_timeout=t, retry_policy=retry)
