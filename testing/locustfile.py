"""
Locust load testing script for the vLLM API server.
"""

from locust import HttpUser, task, between
import json


class ModelAPIUser(HttpUser):
    """
    Locust user class for load testing the model API.
    """
    
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between requests
    
    def on_start(self):
        """
        Called when a simulated user starts.
        """
        self.headers = {
            "Content-Type": "application/json"
        }
    
    @task(3)
    def chat_completion(self):
        """
        Test chat completion endpoint.
        """
        payload = {
            "model": "qwen_lora_sft",
            "messages": [
                {"role": "user", "content": "What is machine learning?"}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def simple_completion(self):
        """
        Test simple completion endpoint.
        """
        payload = {
            "model": "qwen_lora_sft",
            "prompt": "Explain the concept of",
            "temperature": 0.7,
            "max_tokens": 256
        }
        
        with self.client.post(
            "/v1/completions",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def models_list(self):
        """
        Test models list endpoint.
        """
        with self.client.get("/v1/models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
