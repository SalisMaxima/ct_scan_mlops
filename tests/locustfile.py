from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

IMAGE_PATH = Path(__file__).parent / "assets" / "sample.png"


class ApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(2)
    def predict(self):
        with IMAGE_PATH.open("rb") as f:
            files = {"file": ("sample.png", f, "image/png")}
            self.client.post("/predict", files=files)


class StressTestShape(LoadTestShape):
    """Step-wise stress test to find saturation point."""

    stages = [
        {"duration": 20, "users": 100, "spawn_rate": 50},
        {"duration": 40, "users": 250, "spawn_rate": 100},
        {"duration": 60, "users": 400, "spawn_rate": 150},
        {"duration": 80, "users": 600, "spawn_rate": 200},
        {"duration": 100, "users": 800, "spawn_rate": 250},
        {"duration": 120, "users": 0, "spawn_rate": 250},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
