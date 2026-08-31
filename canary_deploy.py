import logging
import random
import time
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CanaryDeployer:
    """
    Phase 10: Simulates A/B routing / Canary Deployments
    In a real Kubernetes environment, this would manipulate Istio/Envoy traffic weights.
    Here we simulate tracking metrics on canary vs stable over time and auto-rollback.
    """
    def __init__(self, canary_weight=0.1, max_error_rate=0.05):
        self.canary_weight = canary_weight
        self.max_error_rate = max_error_rate
        self.canary_calls = 0
        self.canary_errors = 0
        self.stable_calls = 0
        self.stable_errors = 0

    def route_request(self) -> str:
        """Route request probabilistically"""
        return "canary" if random.random() < self.canary_weight else "stable"
        
    def log_result(self, route: str, success: bool):
        if route == "canary":
            self.canary_calls += 1
            if not success:
                self.canary_errors += 1
        else:
            self.stable_calls += 1
            if not success:
                self.stable_errors += 1
                
    def check_health(self) -> bool:
        """Returns False if canary should be rolled back"""
        if self.canary_calls < 50: # Need sample size
            return True
            
        error_rate = self.canary_errors / self.canary_calls
        stable_rate = self.stable_errors / max(self.stable_calls, 1)
        
        logging.info(f"Health Check - Canary Err: {error_rate:.2%}, Stable Err: {stable_rate:.2%}")
        
        # Rollback if canary error rate exceeds threshold AND is worse than stable
        if error_rate > self.max_error_rate and error_rate > (stable_rate * 1.5):
            logging.error(f"Canary failure! Rolling back. Err rate: {error_rate:.2%}")
            return False
        return True
        
    def increase_traffic(self, increment=0.1):
        if self.canary_weight < 1.0:
            self.canary_weight = min(1.0, self.canary_weight + increment)
            logging.info(f"Promoting canary weight to {self.canary_weight:.0%}")

def simulate_canary_deployment():
    deployer = CanaryDeployer(canary_weight=0.10)
    
    # Simulate 500 requests over time
    for i in range(1, 501):
        route = deployer.route_request()
        
        # Simulate request success (Canary has a 2% failure rate, Stable has 1%)
        failure_prob = 0.02 if route == "canary" else 0.01
        success = random.random() > failure_prob
        
        deployer.log_result(route, success)
        
        # Every 100 requests, check health and maybe increase traffic
        if i % 100 == 0:
            if not deployer.check_health():
                logging.info("Initiating Rollback via CI/CD hooks...")
                break
            else:
                deployer.increase_traffic(0.20)
                
if __name__ == "__main__":
    simulate_canary_deployment()
