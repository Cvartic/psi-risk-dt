# Ψ-Risk-DT — Trial SPARQL endpoint

A self-contained Apache Jena Fuseki endpoint hosting the Ψ-Risk-DT
ontology and a small instance graph.

## Bring the endpoint up

From the project root:

```bash
docker compose -f docker/docker-compose.yml up -d fuseki
```

Then point a browser at <http://localhost:3030/>.  The dataset is
`/psi`, queryable at <http://localhost:3030/psi/query>.  Admin login
(only needed for the dashboard, the query endpoint is open) is

```
user:     admin
password: psi-admin     # set in docker/docker-compose.yml
```

To bring it down:

```bash
docker compose -f docker/docker-compose.yml down
```

The dataset is in-memory, so a restart reloads
`psi_ontology.ttl` + `example_data.ttl` from `/staging/`.  Triples
written via SPARQL UPDATE are wiped on restart — intentional for a
trial endpoint.

## Run a query from the CLI

```bash
curl -s --data-urlencode "query@queries/q3_gate_fired.rq" \
     http://localhost:3030/psi/query \
     -H "Accept: application/sparql-results+json" | jq
```

## Local smoke test (no Docker)

The same Turtle is also loadable with rdflib, which is useful for CI and
for catching syntax errors before bringing Fuseki up:

```bash
pip install rdflib
python sparql/smoke_test.py
```

Expected output: ~300 triples loaded; Q1 returns 2 rows, Q2 returns 2
rows, Q3 returns 2 rows.

## What the example dataset contains

One scenario (`escalation_attack` / `mid`) with three sliding windows
that span the three phases of the worked example in §5.6:

| windowId | tStart  | tEnd    | nPackets | phase       | exceedsTauS | neural-only | neurosymbolic |
|---------:|--------:|--------:|---------:|-------------|:-----------:|------------:|--------------:|
|        3 |  15.000 |  20.000 |       24 | stealth     | false       | 0.09        | 0.11          |
|       19 |  47.556 |  52.556 |      151 | ramp-up     | true        | 0.46        | **0.78** ✱   |
|       27 |  87.556 |  92.556 |     4602 | saturation  | true        | **0.91** ✱  | **0.95** ✱   |

✱ indicates `activated = true` for that RiskScore — the symbolic layer
fired and an `Explanation` + `MitigationHint` are attached.  Window 19
mirrors the worked example of §5.6 exactly (151 packets, ≈6× stealth
baseline of 5 pps, ramp-up onset at t=47.556 s).

