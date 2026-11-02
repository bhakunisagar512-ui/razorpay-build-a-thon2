"""Temporal worker entrypoint. The product runs ONE workflow, RevenueRecoveryWorkflow, on the revenue task queue.
"""
import asyncio

from temporalio.client import Client

from services.app.revenue.worker import build
from services.app.settings import settings


async def main() -> None:
    client = await Client.connect(settings.temporal_address)
    print(f"worker up on queue {settings.revenue_task_queue} (RevenueRecoveryWorkflow)", flush=True)
    await build(client).run()


if __name__ == "__main__":
    asyncio.run(main())
