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


def apply_alie_attack(
    model: torch.nn.Module,
    global_state_dict: dict,
    num_clients: int,
    num_malicious: int,
    z_max: float | None = None,
) -> None:
    """Aplica o ataque A Little Is Enough (ALIE) com estimativa local."""
    from scipy.stats import norm

    n, m = num_clients, num_malicious

    if z_max is None:
        s = max(1, (n // 2 + 1) - m)
        # Probabilidade acumulada que define o z-score máximo "seguro"
        p = (n - m - s) / max(n - m, 1)
        p = min(max(p, 1e-6), 1 - 1e-6)   # clamp numérico
        z_max = float(norm.ppf(p))

    with torch.no_grad():
        for (name, param), global_param in zip(
            model.named_parameters(), global_state_dict.values()
        ):
            global_param = global_param.to(param.device)
            mu = global_param.mean()
            sigma = global_param.std()
            low = mu - z_max * sigma
            high = mu + z_max * sigma
            
            # Cria um tensor novo zerado e seleciona aleatóriamente valores entre o intervalo low e high para preenche-lo
            param.data = torch.empty_like(param.data).uniform_(low.item(), high.item())


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

    # Guarda o estado global antes de qualquer ataque (usado pelo ALIE)
    global_state_dict = {
        k: v.clone() for k, v in msg.content["arrays"].to_torch_state_dict().items()
    }

    if is_malicious:
        match attack_type:
            case "gaussian":
                noise_std: float = context.run_config.get("gaussian-noise-std", 1.0)
                print(f"[ATTACK] Cliente {partition_id} aplicando ruído gaussiano (noise_std={noise_std})")
                apply_gaussian_noise_attack(model, noise_std=noise_std)
            case "flip":
                top_fraction: float = context.run_config.get("flip-top-fraction", 1.0)
                print(f"[ATTACK] Cliente {partition_id} aplicando sign flip (top_fraction={top_fraction})")
                apply_sign_flip_attack(model, top_fraction=top_fraction)
            case "alie":
                z_max = context.run_config.get("alie-z-max", None)
                print(f"[ATTACK] Cliente {partition_id} aplicando ALIE (z_max={z_max if z_max is not None else 'auto'})")
                apply_alie_attack(
                    model,
                    global_state_dict=global_state_dict,
                    num_clients=num_partitions,
                    num_malicious=num_malicious,
                    z_max=z_max,
                )
        

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
