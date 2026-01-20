from pathlib import Path

from locust import HttpUser, between, task

IMAGE_PATH = Path(__file__).parent / "assets" / "sample.png"


class ApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        with open(IMAGE_PATH, "rb") as f:
            files = {
                "file": ("sample.png", f, "image/png")
            }
            self.client.post("/predict", files=files)
