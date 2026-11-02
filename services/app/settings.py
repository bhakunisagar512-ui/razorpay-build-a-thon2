from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://recovery:recovery@localhost:5432/recovery"
    temporal_address: str = "localhost:7233"
    demo_mode: bool = True                     # the console and the simulated payment pages only run in demo mode
    ui_data_dir: str = "eval/data/phase4_ui"   # prebuilt datasets, uploaded CSVs and automated run results

    # Revenue-risk pipeline: its own Temporal task queue.
    revenue_task_queue: str = "revenue-q"
    revenue_response_timeout_s: int = 900      # how long a workflow waits for ACCEPT / REJECT on one simulated payment request
    # Demo pacing of planned delays: 100000 is about one second per 28 hours of plan time. The simulated clock keeps quiet hours reproducible.
    revenue_time_scale: int = 100000


settings = Settings()
