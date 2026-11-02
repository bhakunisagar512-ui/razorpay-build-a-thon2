"""Worker for the revenue-risk workflow on its own task queue.  python -m services.app.revenue.worker  (revenue queue only)."""
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from services.app.settings import settings
from services.app.workflows import revenue_activities
from services.app.workflows.revenue import RevenueRecoveryWorkflow


def build(client: Client, task_queue: str | None = None) -> Worker:
    return Worker(client, task_queue=task_queue or settings.revenue_task_queue, workflows=[RevenueRecoveryWorkflow], activities=revenue_activities.ACTIVITIES)


async def main() -> None:
    client = await Client.connect(settings.temporal_address)
    print(f"revenue worker up on queue {settings.revenue_task_queue}", flush=True)
    await build(client).run()


if __name__ == "__main__":
    asyncio.run(main())
