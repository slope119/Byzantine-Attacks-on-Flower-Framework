"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def apply_gaussian_noise_attack(model: torch.nn.Module, noise_std: float = 1.0) -> None:
    """Aplica o ataque de ruído gaussiano"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)


def apply_sign_flip_attack(
    model: torch.nn.Module,
    flip_factor: float = -1.0,
    top_fraction: float = 0.2,
) -> None:
    """Aplica o ataque de Sign Flip.

    Se top_fraction < 1.0, inverte o sinal apenas dos pesos com maior magnitude
    (os top_fraction% mais impactantes de cada camada), em vez de todos os pesos.
    """
    with torch.no_grad():
        for param in model.parameters():
            if top_fraction >= 1.0:
                # Comportamento original: inverte todos os pesos
                param.mul_(flip_factor)
            else:
                flat = param.data.view(-1)
                k = max(1, int(flat.numel() * top_fraction))
                # Seleciona os índices dos k maiores pesos em valor absoluto
                _, top_indices = torch.topk(flat.abs(), k)
                flat[top_indices] *= flip_factor


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    attack_type = context.run_config.get("attack-type", "benign")
    malicious_fraction: float = context.run_config.get("malicious-fraction", 0.0)
    num_malicious = int(num_partitions * malicious_fraction)
    is_malicious = partition_id < num_malicious

    if is_malicious:
        match attack_type:
            case "gaussian":
                print(f"[ATTACK] Cliente {partition_id} aplicando ruído gaussiano")
                apply_gaussian_noise_attack(model)
            case "flip":
                print(f"[ATTACK] Cliente {partition_id} aplicando sign flip ")
                apply_sign_flip_attack(model)
        

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
