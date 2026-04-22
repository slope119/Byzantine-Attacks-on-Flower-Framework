"""pytorchexample: A Flower / PyTorch app."""

import csv
import os
import torch
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, MultiKrum, FedTrimmedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

# Accumulate metrics across rounds
_history = {"round": [], "accuracy": [], "loss": []}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    aggregation_strategy: str = context.run_config.get("aggregation-strategy", "fedavg")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize aggregation strategy from config
    match aggregation_strategy:
        case "multikrum":
            num_malicious_nodes: int = context.run_config.get("multikrum-num-malicious", 1)
            strategy = MultiKrum(
                fraction_evaluate=fraction_evaluate,
                num_malicious_nodes=num_malicious_nodes,
            )
        case "fedtrimmedavg":
            strategy = FedTrimmedAvg(fraction_evaluate=fraction_evaluate)
        case _:  # "fedavg" ou qualquer valor não reconhecido
            strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Plot and save learning progression
    _plot_history(_history)

    # Save results to CSV for later comparison
    attack_type = context.run_config.get("attack-type", "benign")
    malicious_fraction = context.run_config.get("malicious-fraction", 0.0)
    _save_results_csv(_history, attack_type, malicious_fraction)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Accumulate metrics for plotting
    _history["round"].append(server_round)
    _history["accuracy"].append(test_acc)
    _history["loss"].append(test_loss)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def _save_results_csv(history: dict, attack_type: str, malicious_fraction: float) -> None:
    """Salva os resultados da execução em um CSV dentro da pasta results/."""
    os.makedirs("results", exist_ok=True)

    attack_names = {
        "benign":   "TreinamentoBenigno",
        "gaussian": "RuídoGaussiano",
        "flip":     "InversãoDeSinal",
        "alie":     "ALIE",
    }

    # Se não houver clientes maliciosos, força o nome benigno independente do attack-type
    if malicious_fraction == 0.0:
        display_name = "TreinamentoBenigno"
    else:
        display_name = attack_names.get(attack_type, attack_type)

    filename = f"results/{display_name}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "accuracy", "loss"])
        for r, acc, loss in zip(history["round"], history["accuracy"], history["loss"]):
            writer.writerow([r, acc, loss])

    print(f"Resultados salvos em {filename}")


def _plot_history(history: dict) -> None:
    """Plot accuracy and loss progression and save to disk."""
    rounds = history["round"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Federated Learning — Progressão do Modelo")

    ax1.plot(rounds, history["accuracy"], marker="o", color="steelblue")
    ax1.set_title("Acurácia Global")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(rounds)
    ax1.grid(True)

    ax2.plot(rounds, history["loss"], marker="o", color="tomato")
    ax2.set_title("Loss Global")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(rounds)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("learning_progression.png", dpi=150)
    print("Gráfico salvo em learning_progression.png")
    plt.close(fig)
