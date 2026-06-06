"""
Local smoke-test for the Ψ-Risk-DT trial endpoint.

Loads ontology + example data into rdflib (same Turtle the Fuseki container
will mount) and runs every .rq under sparql/queries/.  Used to verify the
queries parse and return non-empty rows before pointing the UI at the live
Fuseki endpoint.
"""

from pathlib import Path
from rdflib import Graph

ROOT     = Path(__file__).parent
ONTOLOGY = ROOT / "ontology" / "psi_ontology.ttl"
DATA     = ROOT / "data"     / "example_data.ttl"
QUERIES  = ROOT / "queries"


def load_graph() -> Graph:
    g = Graph()
    g.parse(ONTOLOGY, format="turtle")
    g.parse(DATA,     format="turtle")
    return g


def run_query(g: Graph, qfile: Path) -> None:
    sparql = qfile.read_text(encoding="utf-8")
    print("─" * 72)
    print(f"▶ {qfile.name}")
    print("─" * 72)

    rows = list(g.query(sparql))
    if not rows:
        print("  (no rows)")
        return

    headers = [str(v) for v in rows[0].labels.keys()] if hasattr(rows[0], "labels") \
              else [f"col{i}" for i in range(len(rows[0]))]

    print("  " + " | ".join(headers))
    for r in rows:
        cells = [str(c) if c is not None else "" for c in r]
        cells = [c[:60] + "…" if len(c) > 60 else c for c in cells]
        print("  " + " | ".join(cells))
    print(f"  → {len(rows)} row(s)")


def main() -> None:
    g = load_graph()
    print(f"Loaded {len(g):,} triples from {ONTOLOGY.name} + {DATA.name}\n")

    for qfile in sorted(QUERIES.glob("*.rq")):
        run_query(g, qfile)


if __name__ == "__main__":
    main()
