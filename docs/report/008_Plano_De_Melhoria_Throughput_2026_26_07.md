# Plano De Melhoria Throughput DRM Language Emitter

Data do relatorio: 2026-07-23  
Nome do arquivo solicitado: `008_Plano_De_Melhoria_Throughput_2026_26_07.md`  
Projeto auditado: `drm-language-emitter`  

## 1. Resumo executivo

O gargalo de throughput do DRM nao parece ser um problema simples de dataset, logging ou checkpoint. Esses pontos existem e devem ser melhorados, mas a diferenca principal vem do custo computacional da propria arquitetura operacional atual: o `DRMEmitterModel` processa a sequencia token-a-token em Python, recalcula ou reaproveita geometria em pequenos blocos, acumula listas de tensores por passo, executa naturalizacao com solve low-rank e so depois empilha estados para emitir logits.

Nos runs locais mais recentes:

| Modelo | Parametros | Tokens | Tempo | Throughput final | Melhor val CE |
|---|---:|---:|---:|---:|---:|
| DRM 125M | 125.161.862 | 150.003.712 | 609.050s / 7,05 dias | 246 tok/s | 1,2143 |
| GPT-2 125M | 126.080.640 | 150.003.712 | 3.482s / 58,0 min | 43.075 tok/s | 1,6919 |

No benchmark curto em `docs/benchmarks/bench_125M`, o DRM ja era cerca de 31x mais lento que GPT-2:

| Modelo | Tokens/s medio | Val CE medio | Memoria media |
|---|---:|---:|---:|
| DRM 125M real | 847,8 | 2,2555 | 3.836 MB |
| GPT-2 125M real | 26.178,6 | 2,8902 | 4.317 MB |
| OPT 125M real | 28.752,0 | 2,9252 | 3.017 MB |

No run longo, a diferenca efetiva ficou em aproximadamente 175x contra GPT-2. Isso tambem mostra degradacao: o DRM comecou perto de 466 tok/s e terminou perto de 246 tok/s. Essa queda deve ser investigada separadamente com telemetria de GPU, memoria e tempo por componente.

Minha recomendacao e atacar em tres ondas:

1. Medicao e correcoes de baixo risco: profiler real de treino, `torch.compile` controlado, batch loader mais eficiente, menos sincronizacoes CPU/GPU, logs de memoria e tempo por etapa.
2. Otimizacoes intra-arquiteturais: reduzir solves, cachear geometria com politica treinavel/agenda, recompilar forward em blocos, remover diagnosticos/regulares caros do caminho quente, usar checkpointing seletivo.
3. Mudancas estruturais compativeis com DRM: processar a dinamica em chunks compilaveis, criar modo de geometria amortizada, especializar metricas diagonais/low-rank durante fases do treino, e explorar kernels customizados para o passo DRM.

## 2. Evidencias observadas no codigo

### 2.1 Forward sequencial token-a-token

Arquivo: `src/drm_language_emitter/model.py`

O forward percorre `seq_len` com:

```python
for t in range(seq_len):
    e_t = token_embeddings[:, t]
    for _ in range(self.config.n_flow_steps):
        ...
```

Esse loop roda 512 passos por microbatch no treino 125M. Diferente de GPT-2/OPT, que executam blocos densos altamente otimizados sobre a sequencia inteira, o DRM atual dispara muitas operacoes pequenas e dependentes do estado anterior. Isso reduz ocupacao da GPU e aumenta overhead de Python/autograd.

Isso nao fere a arquitetura DRM por si so, mas exige uma implementacao mais proxima de scan/chunk/kernels se quisermos throughput competitivo.

### 2.2 Os maiores custos sao geometria e campo direcional

Perfis em `docs/benchmarks/bench_125M/drm_125m_real/seed_*/profile.md` mostram:

| Modulo | seed 1 segundos/run | seed 2 segundos/run |
|---|---:|---:|
| `relational_metric` | 0,1516 | 0,1444 |
| `direction_field` | 0,1396 | 0,1303 |
| `dynamics` | 0,0396 | 0,0370 |
| `emitter` | 0,0224 | 0,0210 |
| `metric_solve` | 0,0114 | 0,0111 |

O perfil simples nao replica exatamente o forward completo atual, mas aponta corretamente onde esta o peso: `RelationalMetric` e `DirectionField`.

### 2.3 `RelationalMetric` usa trunk grande e solve em fp32

Arquivo: `src/drm_language_emitter/metric.py`

O `RelationalMetric` tem uma MLP propria `d_state -> hidden_size -> hidden_size`, com `hidden_size=3584` no 125M. Ele tambem produz diagonal e matriz low-rank. A naturalizacao usa Woodbury e desliga autocast:

```python
with torch.autocast(device_type=autocast_device, enabled=False):
    ...
    correction_coeff = torch.linalg.solve(middle, rhs)
```

Esse bloco e numericamente defensavel, mas caro no caminho quente. Como `metric_rank=64`, o solve e pequeno, porem chamado muitas vezes e em fp32. O custo total vem da repeticao por token e por microbatch.

### 2.4 `DirectionField` tambem tem trunk grande e gera muitas direcoes

Arquivo: `src/drm_language_emitter/direction_field.py`

Com `d_state=1536`, `n_directions=64`, `direction_basis_size=128`, o modelo calcula coeficientes e faz `matmul` contra uma base. Isso preserva a ideia DRM de direcoes ativas, mas o custo e alto quando repetido a cada atualizacao geometrica.

### 2.5 A geometria ja e cacheada, mas de forma limitada

Config 125M:

```yaml
geometry_update_interval: 4
direction_basis_size: 128
metric_u_basis_size: 128
bptt_truncate_interval: 64
```

O forward recalcula geometria a cada 4 ticks. Isso ja e uma otimizacao importante, mas ainda significa cerca de 128 recalculos geometricos por sequencia de 512 tokens por microbatch. Cada recalculo chama `direction_field`, `metric` e `risk`.

### 2.6 O loader faz copias Python evitaveis

Arquivo: `src/drm_language_emitter/data.py`

O `MemmapTokenDataset.window` faz:

```python
raw = self._read_range(start, seq_len + 1)
values = torch.tensor(list(raw), dtype=torch.long)
```

Isso cria uma lista Python por janela. Em seguida, `make_batch` empilha as janelas e copia para o device. No GPT-2 esse custo fica escondido pelo throughput alto do modelo, mas no DRM ainda adiciona overhead e variancia. A troca para `torch.frombuffer`, staging CPU pinned e copia assicrona e uma melhoria de baixo risco.

### 2.7 Diagnosticos foram reduzidos no treino memmap, mas perdas auxiliares continuam no caminho quente

`scripts/train_drm_memmap.py` chama:

```python
out = model(x, y, global_step=step, collect_diagnostics=False)
```

Isso evita quantis e diagnosticos caros, mas o modelo ainda acumula `action_values`, `dim_values`, `entropy_values`, `metric_regs`, `condition_values`, `u_norm_values`, etc. Mesmo quando um peso auxiliar e pequeno, seu tensor pode manter grafo e custo de memoria.

## 3. Causas raiz provaveis

### Causa 1: recorrencia estrita por token

O estado `z` no token `t + 1` depende de `z` atualizado no token `t`. Isso impede paralelismo completo no eixo temporal, ao contrario de Transformer. Esta e uma caracteristica arquitetural, nao um bug.

O objetivo entao nao deve ser transformar o DRM em Transformer, mas reduzir o custo de cada passo e compilar/agrupar a recorrencia.

### Causa 2: sub-redes geometricas grandes no caminho quente

`DirectionField` e `RelationalMetric` sao MLPs grandes, chamadas muitas vezes por sequencia. O custo do DRM nao esta no embedding nem principalmente no emissor de logits; esta no calculo da geometria do espaco relacional.

### Causa 3: muitos tensores intermediarios retidos para losses auxiliares

Mesmo sem diagnosticos completos, o forward cria listas por passo e empilha no fim. Isso aumenta memoria, pressao no autograd e overhead Python.

### Causa 4: naturalizacao numericamente robusta, mas cara

A solucao Woodbury fp32 e correta para estabilidade, porem precisa de agenda, aproximacao ou frequencia menor. Hoje ela e parte do passo normal do treino.

### Causa 5: ausencia de profiler de treino real

`scripts/profile_drm.py` e util, mas roda `torch.no_grad()`, CPU por padrao se nao houver checkpoint em CUDA, e nao reproduz grad accumulation, backward, optimizer e autocast do treino real. Precisamos de um perfil de `N` steps reais com `torch.profiler`.

## 4. Alternativas de melhoria sem ferir a arquitetura

### A. Instrumentacao primeiro

**A1. Profiler real de treino DRM**

Criar `scripts/profile_drm_training_step.py` ou estender `train_drm_memmap.py` com `--profile-steps`.

Medir:

- tempo de `make_batch`
- H2D copy
- forward
- backward
- grad clip
- optimizer
- eval/checkpoint
- `torch.cuda.max_memory_allocated`
- `torch.cuda.memory_reserved`
- tokens/s por janela movel, nao apenas acumulado

Risco arquitetural: nenhum.  
Ganho esperado: nao acelera sozinho, mas evita otimizar no escuro.  
Prioridade: P0.

**A2. Log de throughput instantaneo**

Hoje `tokens_per_sec` e acumulado desde o inicio. Isso esconde queda gradual. Adicionar `step_elapsed_sec` e `window_tokens_per_sec` nos ultimos 10/50/100 steps.

Risco: nenhum.  
Prioridade: P0.

**A3. Comparar forward isolado com backward completo**

Medir quanto tempo e forward versus backward. Se backward for muito maior por causa dos tensors acumulados, checkpointing e pruning de losses terao prioridade.

Risco: nenhum.  
Prioridade: P0.

### B. Quick wins no codigo atual

**B1. Ativar `torch.compile` por experimento controlado**

O config tem `use_torch_compile: false`. O modelo ja possui suporte parcial:

```python
self._compiled_forward = torch.compile(self._forward_impl)
```

Mas a implementacao atual compila a funcao inteira, que tem listas, condicionais e `global_step`. Melhor testar:

- `torch.compile(model, mode="reduce-overhead")`
- compilar submodulos (`DirectionField`, `RelationalMetric`, `DRMFlow`, `LanguageEmitter`) primeiro
- compilar uma funcao `drm_step(z, e_t, cached_geometry, ...)`

Risco: baixo a medio; pode falhar ou recompilar demais.  
Ganho esperado: 1,2x a 2x se o overhead Python for relevante; mais se o passo for bem encapsulado.  
Prioridade: P1.

**B2. Remover sincronizacoes CPU/GPU no caminho de treino**

No loop, cada microbatch soma:

```python
step_loss += float(out["aux_losses"].get("ce", out["loss"]).detach().cpu())
```

Esse `float(...cpu())` sincroniza a GPU a cada acumulacao. Em `grad_accum_steps=8`, sao 8 sincronizacoes por step. Guardar o tensor detachado e converter so no log reduz stalls.

Risco: baixo.  
Ganho esperado: pequeno a moderado, dependendo do runtime.  
Prioridade: P1.

**B3. Otimizar `MemmapTokenDataset`**

Trocar `torch.tensor(list(raw))` por caminho sem lista Python. Opcoes:

- `torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()`
- prealocar batch CPU e preencher via NumPy/frombuffer
- usar `pin_memory=True` manual e `.to(device, non_blocking=True)`
- opcionalmente criar worker prefetch simples em thread/processo

Risco: baixo, com testes em `tests/test_token_shards.py`.  
Ganho esperado: pequeno no gargalo atual, mas importante para estabilidade e multi-GPU.  
Prioridade: P1.

**B4. Usar `optimizer.zero_grad(set_to_none=True)` ja existe; manter**

Esse ponto esta correto no script. Nao ha acao aqui alem de preservar.

**B5. Testar fused AdamW**

Em CUDA moderna:

```python
torch.optim.AdamW(..., fused=True)
```

Adicionar flag `--fused-adamw` com fallback se nao suportado.

Risco: baixo.  
Ganho esperado: pequeno a moderado; optimizer provavelmente nao e o gargalo principal.  
Prioridade: P2.

### C. Reduzir custo das perdas auxiliares

**C1. Calcular losses auxiliares em stride**

Hoje `action_loss`, `dim_sparsity`, `entropy`, `metric_reg`, `condition` e `u_norm` sao coletadas ao longo de todos os passos. Alternativa:

- calcular CE em todos os tokens
- calcular losses geometricas apenas a cada `aux_loss_interval`
- manter `aux_loss_interval=1` como modo fiel atual
- testar `4`, `8`, `16`

Isso preserva a arquitetura: a dinamica continua DRM, mas a supervisao geometrica nao precisa ser densa em todo token.

Risco: medio; pode afetar qualidade da geometria.  
Ganho esperado: moderado, principalmente em memoria/autograd.  
Prioridade: P1.

**C2. Desacoplar metric diversity do treino base**

`lambda_metric_diversity=0.001` faz o forward manter `metric_diag_steps` quando o peso e diferente de zero. Testar agenda:

- desligar nos primeiros runs de throughput
- aplicar so em steps de diagnostico
- calcular em mini-batches menores

Risco: medio para propriedades geometricas; baixo para CE.  
Ganho esperado: pequeno a moderado.  
Prioridade: P1.

**C3. Losses com `detach` parcial**

Algumas metricas auxiliares podem orientar logging sem gradient. Separar:

- losses que realmente treinam geometria
- diagnosticos sem grad
- proxies de estabilidade/condicao em `no_grad` quando peso zero

Risco: baixo se preservado por flags.  
Ganho esperado: moderado se reduzir grafo retido.  
Prioridade: P1.

### D. Geometria amortizada

**D1. Aumentar `geometry_update_interval` com agenda**

Hoje o 125M usa `geometry_update_interval: 4`. Testar:

- warmup: 1 ou 2 nos primeiros steps
- treino principal: 8, 16, 32
- fine-tune final: voltar para 4

A arquitetura continua DRM porque a trajetoria ainda usa direcoes e metrica; apenas a geometria e tratada como campo quasi-estatico por mais tokens.

Risco: medio; pode perder responsividade local.  
Ganho esperado: alto. Se `DirectionField + RelationalMetric` dominam o custo, passar de 4 para 16 pode reduzir grande parte do gasto geometrico.  
Prioridade: P1.

**D2. Congelar ou atualizar parcialmente geometria em fases**

Treinar em fases:

1. emissor + dinamica com geometria menos frequente
2. descongelar/fortalecer metrica
3. fine-tune com metrica completa

Risco: medio.  
Ganho esperado: alto para pretreino longo.  
Prioridade: P2.

**D3. Campo direcional baseado em delta de estado**

Recalcular geometria quando `||z - z_cached||` ultrapassar limite, nao por intervalo fixo. Isso pode reduzir recalculos em trechos estaveis.

Risco: medio; branching dinamico pode atrapalhar compile.  
Ganho esperado: medio.  
Prioridade: P3.

### E. Naturalizacao e metrica

**E1. Agenda de naturalizacao**

Config atual:

```yaml
use_metric_naturalization: true
metric_naturalization_strength: 0.5
metric_naturalization_warmup_steps: 500
metric_damping: 0.3
metric_rank: 64
```

Alternativas:

- `metric_naturalization_strength=0` no warmup de linguagem
- ligar gradualmente depois de CE estabilizar
- aplicar naturalizacao a cada `metric_solve_interval`, reaproveitando `inv_diag`/low-rank entre steps
- usar metrica diagonal durante pretreino e low-rank no fine-tune

Risco: medio a alto para a identidade matematica do DRM se removido permanentemente; baixo se for agenda/fase.  
Ganho esperado: moderado a alto.  
Prioridade: P1/P2.

**E2. Reduzir `metric_rank` em pretreino**

Testar ranks `0`, `8`, `16`, `32`, `64`. Rank 0 vira metrica diagonal, ainda relacional pelo campo/dinamica, mas perde low-rank coupling. Uma opcao conservadora:

- pretreino rank 8/16
- fine-tune rank 64

Risco: medio.  
Ganho esperado: medio, alem de simplificar solve.  
Prioridade: P2.

**E3. Compartilhar trunk entre `DirectionField` e `RelationalMetric`**

Hoje ambos tem trunks MLP separados e grandes sobre o mesmo `z`. Uma alternativa compativel:

- criar `GeometryEncoder(z)` compartilhado
- `DirectionField` e `RelationalMetric` viram heads
- preservar saidas: direcoes, gates, diag, U

Isso altera parametrizacao, mas nao altera a arquitetura conceitual. Pode inclusive reduzir parametros ou redistribui-los para heads.

Risco: medio; checkpoint antigo nao carrega diretamente.  
Ganho esperado: alto, pois elimina uma das duas MLPs grandes por atualizacao geometrica.  
Prioridade: P2.

**E4. Low-rank basis menor**

O 125M usa `direction_basis_size=128` e `metric_u_basis_size=128`. Testar 32/64/128 com mesma contagem aproximada de parametros realocada ao emissor ou dinamica.

Risco: medio.  
Ganho esperado: medio.  
Prioridade: P2.

### F. Chunking e compilacao do passo recorrente

**F1. Criar `DRMStep` explicito**

Extrair o miolo:

```python
z, per_step_stats = drm_step(z, e_t, cached_geometry, naturalization_strength)
```

Vantagens:

- melhor alvo para `torch.compile`
- menor funcao para testar
- possibilidade futura de kernel customizado
- reduz complexidade do forward

Risco: baixo se comportamento for coberto por teste de equivalencia.  
Ganho esperado: indireto, mas necessario para ganhos maiores.  
Prioridade: P1.

**F2. Processar chunks de sequencia**

Criar `forward_chunked(input_ids, chunk_size=32/64)`:

- token embeddings para chunk
- loop interno compilado
- emission states por chunk
- truncamento BPTT alinhado ao chunk

Isso nao paraleliza completamente o tempo, mas reduz overhead e melhora compile.

Risco: medio.  
Ganho esperado: medio a alto.  
Prioridade: P2.

**F3. Usar `torch.func`/scan quando estavel**

Se a versao local do PyTorch suportar um scan eficiente, reescrever a recorrencia como scan. Caso contrario, manter chunk + compile.

Risco: medio a alto por maturidade/flexibilidade.  
Ganho esperado: alto se funcionar.  
Prioridade: P3.

### G. Checkpointing e memoria

**G1. Activation checkpointing seletivo**

Usar `torch.utils.checkpoint` em `direction_field`, `metric` ou no chunk. Isso troca compute por memoria. Pode parecer contraintuitivo para throughput, mas se a degradacao de 466 para 246 tok/s vier de pressao de memoria, fragmentacao ou paging, pode estabilizar.

Risco: medio; pode reduzir throughput se compute ja domina.  
Ganho esperado: incerto, deve ser guiado por profiler.  
Prioridade: P2.

**G2. Reduzir retencao de listas**

Hoje listas como `action_values`, `dim_values`, `metric_regs`, `condition_values` acumulam tensores por passo. Alternativa:

- acumuladores online: `sum_action += energy.mean()`
- evitar `torch.stack` quando so precisamos da media
- guardar series apenas se diagnostico/loss precisar

Risco: baixo a medio; requer cuidado para manter grad correto.  
Ganho esperado: medio em memoria e overhead.  
Prioridade: P1.

### H. Multi-GPU e escala

**H1. DDP ja existe, mas o batch efetivo precisa subir**

`train_drm_memmap.py` tem DDP. Como o DRM tem baixo uso de memoria comparado ao GPT-2 nos benchmarks, pode haver espaco para batch maior por GPU. Antes de multi-GPU, testar:

- `batch_size=4`, `grad_accum=4`
- `batch_size=8`, `grad_accum=2`
- manter tokens_per_step semelhante para comparar throughput

Risco: baixo.  
Ganho esperado: medio se a GPU estiver subutilizada por microbatches pequenos.  
Prioridade: P1.

**H2. Pipeline/model parallel nao e prioridade**

O modelo cabe em uma 4090. O problema principal nao e memoria de parametros, e sim passo recorrente pequeno e caro. Multi-GPU ajuda wall clock, mas nao resolve eficiencia por GPU.

Risco: alto em complexidade.  
Prioridade: P4.

### I. Aproximacoes arquiteturais conservadoras

**I1. Modo `fast_pretrain`**

Adicionar config explicito:

```yaml
fast_pretrain:
  geometry_update_interval: 16
  aux_loss_interval: 8
  metric_rank_train: 16
  naturalization_interval: 4
```

Depois fine-tune com modo completo. Isso preserva a arquitetura final e permite estudar se a qualidade linguistica vem mais da dinamica global do que da geometria densa a cada token.

Risco: medio.  
Ganho esperado: alto.  
Prioridade: P1/P2.

**I2. Destilacao temporal interna**

Treinar um pequeno preditor de geometria para interpolar entre recalculos completos:

- geometria completa a cada N tokens
- preditor leve atualiza gates/diag entre recalculos

Risco: alto, mas ainda compativel com DRM.  
Ganho esperado: alto se bem sucedido.  
Prioridade: P4.

## 5. Experimentos recomendados

### Experimento 0: baseline reproduzivel

Objetivo: confirmar numeros em run curto de 5M ou 10M tokens.

Manter:

- `configs/drm_125m_4090.yaml`
- `batch_size=2`
- `grad_accum_steps=8`
- `seq_len=512`
- `precision=bf16`

Registrar:

- throughput instantaneo
- forward/backward/optimizer/data
- memoria
- temperatura/clocks se possivel fora do Python

### Experimento 1: loader e sincronizacao

Mudancas:

- remover `.cpu()` por microbatch
- usar loader sem `list(raw)`
- H2D non-blocking com pinned memory

Aceite:

- mesma CE em smoke test
- nenhum drift em batch deterministico
- throughput melhora ou permanece igual

### Experimento 2: geometria interval

Grid:

| `geometry_update_interval` | `aux_loss_interval` | `metric_rank` | Observacao |
|---:|---:|---:|---|
| 4 | 1 | 64 | baseline |
| 8 | 1 | 64 | conservador |
| 16 | 1 | 64 | amortizado |
| 16 | 4 | 64 | amortizado + aux sparse |
| 32 | 8 | 64 | agressivo |

Medir CE, estabilidade de geracao e throughput.

### Experimento 3: naturalizacao

Grid:

| Config | Esperado |
|---|---|
| naturalizacao atual | qualidade atual |
| strength 0 ate 25M tokens, depois 0.5 | pretreino mais rapido |
| solve a cada 4 tokens | menos solve |
| rank 16 no pretreino, rank 64 fine-tune | menor custo inicial |

### Experimento 4: trunk compartilhado

Implementar `GeometryEncoder` compartilhado em branch separada. Testar contagem de parametros proxima a 125M.

Aceite:

- forward equivalente em shapes
- smoke tests
- run 2M tokens comparavel
- perfil mostra reducao clara de `direction_field + relational_metric`

### Experimento 5: `DRMStep` compilado

Extrair passo, criar teste de equivalencia com forward antigo e ativar:

- `torch.compile(step_fn, mode="reduce-overhead")`
- `torch.compile(model, fullgraph=False)`
- chunk sizes 16/32/64

Aceite:

- sem recompilacoes repetidas por `global_step`
- throughput melhora em run de pelo menos 1.000 steps

## 6. Mudancas de codigo sugeridas

### 6.1 `src/drm_language_emitter/data.py`

Prioridade:

- substituir `torch.tensor(list(raw))`
- suportar batch CPU pinned
- adicionar teste deterministico para shard crossing

### 6.2 `scripts/train_drm_memmap.py`

Prioridade:

- log de throughput instantaneo
- evitar `.cpu()` por acumulacao
- adicionar profiler real via flag
- opcao `--fused-adamw`
- opcao `--compile-mode`
- opcao `--profile-output`

### 6.3 `src/drm_language_emitter/model.py`

Prioridade:

- acumuladores online para losses
- `aux_loss_interval`
- `naturalization_interval`
- `DRMStep` separado
- `forward_chunked`
- reduzir tensors retidos quando pesos auxiliares sao zero

### 6.4 `src/drm_language_emitter/config.py`

Adicionar campos:

```python
aux_loss_interval: int = 1
naturalization_interval: int = 1
compile_submodules: bool = False
fast_pretrain: bool = False
```

Ou manter configs explicitos sem campo `fast_pretrain`, para evitar semantica magica.

### 6.5 `src/drm_language_emitter/metric.py`

Prioridade:

- permitir `naturalize_mode`: `full`, `diag`, `none`, `interval`
- expor caminho diagonal rapido
- avaliar solve batched com `cholesky_solve` para rank pequeno, se numericamente melhor no hardware

### 6.6 `src/drm_language_emitter/direction_field.py` e `metric.py`

Prioridade futura:

- compartilhar trunk via `GeometryEncoder`
- manter heads separados
- preservar configs antigos com fallback

## 7. Riscos tecnicos

### Risco: ganhar throughput e perder a vantagem linguistica

O DRM venceu em CE e qualidade subjetiva no run de 150M tokens. Otimizacoes que reduzem geometria demais podem aproximar o modelo de um RNN/MLP simples. Por isso os experimentos devem sempre medir:

- CE train/val
- geracao qualitativa fixa por prompts
- diagnosticos de gates/dimD
- estabilidade de sequencia

### Risco: `torch.compile` mascarar regressao ou recompilar

Loops com condicionais, listas e `global_step` podem causar recompilacoes. O teste deve registrar tempo apos warmup e, se possivel, logs de recompilacao.

### Risco: o run longo degradar por fator externo

A queda de throughput de 466 para 246 tok/s pode vir de:

- clocks/temperatura/power limit
- fragmentacao de memoria
- crescimento de historico JSON
- antivirus/IO no Windows
- checkpoint/eval
- sincronizacoes frequentes

O profiler real precisa separar essas hipoteses.

## 8. Ordem recomendada de implementacao

1. Adicionar profiler real, throughput instantaneo e logs de memoria.
2. Remover sincronizacao `.cpu()` por microbatch.
3. Otimizar `MemmapTokenDataset`.
4. Trocar acumulacao por listas para acumuladores online quando possivel.
5. Adicionar `aux_loss_interval`.
6. Testar `geometry_update_interval` 8/16/32.
7. Adicionar `naturalization_interval` e agenda de naturalizacao.
8. Extrair `DRMStep` e testar `torch.compile`.
9. Implementar `forward_chunked`.
10. Explorar `GeometryEncoder` compartilhado.

## 9. Meta realista de throughput

Nao e realista esperar que o DRM recorrente token-a-token alcance GPT-2 sem mudancas profundas de kernel/scan. Mas ha uma meta plausivel em etapas:

| Fase | Meta |
|---|---:|
| Baseline atual run longo | 246 tok/s |
| Quick wins + loader + sync | 300-600 tok/s |
| Aux sparse + geometria 16 | 800-2.000 tok/s |
| Step compilado/chunked | 1.500-4.000 tok/s |
| Trunk compartilhado + agenda metric | 3.000-8.000 tok/s |
| Kernel/scan customizado | 8.000+ tok/s, incerto |

Mesmo 3.000 tok/s reduziria 150M tokens de 7,05 dias para cerca de 13,9 horas. Em 8.000 tok/s, o mesmo treino cairia para cerca de 5,2 horas.

## 10. Conclusao

O DRM esta pagando caro pela propriedade que parece ter dado vantagem no resultado: uma dinamica relacional recorrente com geometria ativa. O caminho correto nao e remover essa dinamica, e sim amortizar, compilar e medir melhor o passo DRM.

As alternativas mais alinhadas com a arquitetura sao:

- geometria amortizada por intervalo/agenda
- perdas geometricas em stride
- naturalizacao agendada
- acumuladores online
- loader e treino sem sincronizacao desnecessaria
- `DRMStep` compilado em chunks
- trunk geometrico compartilhado

O primeiro milestone deve ser tecnico e mensuravel: levar o run curto 125M de menos de 1k tok/s para pelo menos 2k tok/s sem degradar muito a CE. Depois disso faz sentido buscar a faixa de 5k-8k tok/s com mudancas estruturais mais fortes.
