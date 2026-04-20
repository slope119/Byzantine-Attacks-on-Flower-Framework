# Descrição Técnica dos Ataques em Aprendizado Federado

## Visão Geral

Este projeto implementa ataques Byzantine em um cenário de aprendizado federado utilizando o framework [Flower (flwr)](https://flower.ai/). Os ataques são aplicados no lado do cliente (`client_app.py`) após o treinamento local, antes do envio dos pesos ao servidor agregador.

---

## Seleção de Clientes Maliciosos

A seleção de quais clientes se comportam de forma maliciosa é feita de maneira determinística com base no `partition-id` de cada cliente e na fração de clientes maliciosos configurada.

```python
num_malicious = int(num_partitions * malicious_fraction)
is_malicious = partition_id < num_malicious
```

Os clientes com os menores `partition-id`s são designados como maliciosos. Por exemplo, com 10 clientes e `malicious-fraction = 0.4`, os clientes de id `0` a `3` são maliciosos. Essa abordagem é determinística e reproduzível, facilitando experimentos controlados.

---

## Configuração via `pyproject.toml`

Os parâmetros de ataque são passados ao cliente através da seção `[tool.flwr.app.config]` do `pyproject.toml`, sendo lidos em tempo de execução via `context.run_config`. Os parâmetros relevantes são:

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `attack-type` | `string` | Tipo de ataque: `"benign"`, `"gaussian"`, `"flip"` ou `"alie"` |
| `malicious-fraction` | `float` | Fração de clientes que aplicam o ataque (ex: `0.4` = 40%) |

Exemplo de configuração:

```toml
[tool.flwr.app.config]
attack-type = "alie"
malicious-fraction = 0.4
```

---

## Ataques Implementados

### 1. Ruído Gaussiano (`gaussian`)

Adiciona ruído amostrado de uma distribuição normal a cada parâmetro do modelo treinado.

```
w_attack = w_local + N(0, noise_std)
```

**Ferramenta:** `torch.randn_like` para geração do ruído com a mesma forma e dispositivo do tensor original.

**Parâmetro:** `noise_std` (desvio padrão do ruído, padrão `1.0`) — configurável diretamente no código.

---

### 2. Sign Flip (`flip`)

Inverte o sinal dos pesos do modelo treinado. Suporta dois modos:

- **Modo completo** (`flip-top-fraction = 1.0`): inverte o sinal de todos os pesos.
- **Modo seletivo** (`flip-top-fraction < 1.0`): inverte o sinal apenas dos `k` pesos com maior magnitude absoluta em cada camada, onde `k = floor(numel * top_fraction)`.

```
# Modo seletivo
flat = param.view(-1)
k = floor(numel * top_fraction)
top_indices = argtopk(|flat|, k)
flat[top_indices] *= -1
```

**Ferramenta:** `torch.topk` para seleção eficiente dos `k` maiores valores em valor absoluto.

**Parâmetro:** `flip-top-fraction` — controla a agressividade do ataque. Valor `1.0` corresponde ao Sign Flip original; valores menores tornam o ataque mais sutil.

---

### 3. A Little Is Enough — ALIE (`alie`)

Implementação do ataque proposto por Baruch et al. (2019), adaptado para operar sem acesso aos pesos de outros clientes (estimativa local).

#### Estimativas locais

Como o cliente malicioso não tem acesso aos updates dos demais clientes, as estatísticas populacionais são aproximadas localmente:

- **Média (μ):** pesos do modelo global recebido no início do round, antes do treinamento local. Representa a melhor estimativa disponível da média dos clientes benignos.
- **Desvio padrão (σ):** diferença absoluta entre os pesos treinados localmente e o modelo global, capturando o quanto um cliente típico se afasta da média após uma rodada de treinamento.

```
μ = w_global
σ = |w_local - w_global|
```

#### Perturbação

O vetor de ataque é construído deslocando a média na direção de maior variância:

```
w_attack = μ + z_max * σ
```

#### Cálculo automático do z_max

O `z_max` é o maior z-score que ainda não é identificado como outlier pelo agregador. É calculado a partir da fórmula do paper original:

```
s     = floor(n/2 + 1) - m
p     = (n - m - s) / (n - m)
z_max = Φ⁻¹(p)
```

Onde:
- `n` = número total de clientes
- `m` = número de clientes maliciosos
- `Φ⁻¹` = função quantil (inversa da CDF) da distribuição normal padrão

**Ferramenta:** `scipy.stats.norm.ppf` para o cálculo de `Φ⁻¹(p)`.

O `z_max` cresce com a proporção de clientes maliciosos. Exemplos com `n = 20`:

| `malicious-fraction` | `m` | `z_max` |
|---|---|---|
| 0.2 | 4 | ≈ -0.43 (ataque ineficaz) |
| 0.3 | 6 | ≈ 0.00 (limiar) |
| 0.4 | 8 | ≈ 0.67 |
| 0.5 | 10 | ≈ 1.28 |

O ataque só produz efeito positivo quando `m > n/4`. Abaixo desse limiar, o `z_max` é negativo e a perturbação empurra os pesos na direção oposta à média, com impacto mínimo.

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `torch` | Manipulação de tensores, operações in-place nos pesos do modelo |
| `scipy>=1.11.0` | Cálculo do quantil da normal padrão (`norm.ppf`) no ataque ALIE |
| `flwr[simulation]>=1.28.0` | Framework de aprendizado federado (comunicação, agregação, contexto) |
