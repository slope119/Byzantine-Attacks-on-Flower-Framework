"""
Script para plotar a comparação entre múltiplas execuções de treinamento federado.

# Plota todos os CSVs da pasta results/ automaticamente
python plot_comparison.py

# Ou escolhe arquivos específicos
python plot_comparison.py results/alie_*.csv results/benign_*.csv

# Muda o nome do arquivo de saída
python plot_comparison.py --output meu_grafico.png

"""


import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt


def load_csv(filepath: str) -> dict:
    """Carrega um CSV de resultados e retorna um dicionário com rounds, accuracy e loss."""
    rounds, accuracy, loss = [], [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            accuracy.append(float(row["accuracy"]))
            loss.append(float(row["loss"]))
    return {"rounds": rounds, "accuracy": accuracy, "loss": loss}


def label_from_filename(filepath: str) -> str:
    """Gera um label legível a partir do nome do arquivo CSV.

    Exemplos:
        results/ALIE.csv               → ALIE
        results/RuídoGaussiano.csv     → RuídoGaussiano
        results/InversãoDeSinal.csv    → InversãoDeSinal
        results/TreinamentoBenigno.csv → TreinamentoBenigno
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def plot_comparison(filepaths: list[str], output: str) -> None:
    """Plota accuracy e loss de múltiplas execuções no mesmo gráfico."""
    if not filepaths:
        print("Nenhum arquivo CSV encontrado.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_rounds = []

    for i, filepath in enumerate(sorted(filepaths)):
        data = load_csv(filepath)
        label = label_from_filename(filepath)
        color = colors[i % len(colors)]

        ax1.plot(data["rounds"], data["accuracy"], marker="o", label=label, color=color)
        ax2.plot(data["rounds"], data["loss"], marker="o", label=label, color=color)
        all_rounds = data["rounds"]  # todos os CSVs têm os mesmos rounds

    even_rounds = [r for r in all_rounds if r % 2 == 0]
    ax1.set_xticks(even_rounds)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax2.set_xticks(even_rounds)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))

    for ax, ylabel in [(ax1, "Acurácia"), (ax2, "Perda")]:
        ax.set_xlabel("Rodada", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=13)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output, format="pdf")
    print(f"Gráfico de comparação salvo em {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plota comparação entre execuções federadas.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Arquivos CSV a incluir no gráfico. Se omitido, usa todos em results/",
    )
    parser.add_argument(
        "--output",
        default="comparison.pdf",
        help="Nome do arquivo de saída (padrão: comparison.pdf)",
    )
    args = parser.parse_args()

    if args.files:
        filepaths = args.files
    else:
        results_dir = "results"
        if not os.path.isdir(results_dir):
            print(f"Pasta '{results_dir}' não encontrada. Execute ao menos uma simulação primeiro.")
            sys.exit(1)
        filepaths = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]

    plot_comparison(filepaths, args.output)


if __name__ == "__main__":
    main()
