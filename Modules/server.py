import flwr as fl
from flwr.server.strategy.fedavg import FedAvg

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    strategy = FedAvg(min_fit_clients=2, min_eval_clients=2, min_available_clients=2)
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 10})
