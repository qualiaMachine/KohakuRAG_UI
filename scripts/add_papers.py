#!/usr/bin/env python3
"""Add new papers to metadata.csv for corpus expansion.

Usage:
    python scripts/add_papers.py              # Dry-run: show what would be added
    python scripts/add_papers.py --apply      # Actually append to metadata.csv

After running with --apply, rebuild the index:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
"""

import csv
import sys
from pathlib import Path

METADATA_CSV = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"

# ============================================================================
# NEW PAPERS TO ADD
# Format: (id, type, title, year, citation, url)
# ============================================================================

NEW_PAPERS = [
    # --- Water usage / footprint ---
    (
        "ren2024",
        "paper",
        "Reconciling the Contrasting Narratives on the Environmental Impact of Large Language Models",
        "2024",
        "Shaolei Ren, Bill Tomlinson, Rebecca W. Black, A. Torrance (2024). Reconciling the contrasting narratives on the environmental impact of large language models. Scientific Reports. https://arxiv.org/pdf/2409.07116",
        "https://arxiv.org/pdf/2409.07116",
    ),
    (
        "li2024_water",
        "paper",
        "Towards Sustainable GenAI using Generation Directives for Carbon-Friendly Large Language Model Inference",
        "2024",
        "Baolin Li, Yankai Jiang, Vijay Gadepally, Devesh Tiwari (2024). Towards Sustainable GenAI using Generation Directives for Carbon-Friendly Large Language Model Inference. arXiv. https://arxiv.org/pdf/2403.12900",
        "https://arxiv.org/pdf/2403.12900",
    ),
    # --- Datacenter sustainability ---
    (
        "acun2023",
        "paper",
        "Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters",
        "2023",
        "Bilge Acun, Benjamin Lee, Fiodar Kazhamiaka, Kiwan Maeng, Udit Gupta, Manoj Chakkaravarthy, David Brooks, Carole-Jean Wu (2023). Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters. ASPLOS '23. https://arxiv.org/pdf/2210.02681",
        "https://arxiv.org/pdf/2210.02681",
    ),
    (
        "radovanovic2022",
        "paper",
        "Carbon-Aware Computing for Datacenters",
        "2022",
        "Ana Radovanovic, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Duber, Binz Roy, David Talaber, Drew Ferguson, Nic Tills, Kathy Zhu, Max Nova, Jared Chen, Ken Hua (2022). Carbon-Aware Computing for Datacenters. IEEE TPDS. https://arxiv.org/pdf/2106.11750",
        "https://arxiv.org/pdf/2106.11750",
    ),
    # --- GPU / hardware energy efficiency ---
    (
        "desislavov2023",
        "paper",
        "Trends in AI Inference Energy Consumption: Beyond the Performance-vs-Parameter Laws of Deep Learning",
        "2023",
        "Radosvet Desislavov, Fernando Martinez-Plumed, Jose Hernandez-Orallo (2023). Trends in AI Inference Energy Consumption: Beyond the Performance-vs-Parameter Laws of Deep Learning. Sustainable Computing. https://arxiv.org/pdf/2301.00774",
        "https://arxiv.org/pdf/2301.00774",
    ),
    (
        "samsi2023_gpu",
        "paper",
        "Benchmarking Large Language Models on Supercomputers",
        "2023",
        "Siddharth Samsi, Dan Zhao, Andrew Gittens, David Bader, Vijay Gadepally (2023). Benchmarking Large Language Models on Supercomputers. arXiv. https://arxiv.org/pdf/2402.05065",
        "https://arxiv.org/pdf/2402.05065",
    ),
    # --- Inference optimization for sustainability ---
    (
        "xu2024",
        "paper",
        "A Survey on Model Compression for Large Language Models",
        "2024",
        "Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang (2024). A Survey on Model Compression for Large Language Models. TACL. https://arxiv.org/pdf/2308.07633",
        "https://arxiv.org/pdf/2308.07633",
    ),
    (
        "stojkovic2024",
        "paper",
        "Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference",
        "2024",
        "Jovan Stojkovic, Esha Choukse, Chaojie Zhang, Inigo Goiri, Josep Torrellas (2024). Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference. arXiv. https://arxiv.org/pdf/2403.20306",
        "https://arxiv.org/pdf/2403.20306",
    ),
    (
        "chavan2024",
        "paper",
        "Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward",
        "2024",
        "Arnav Chavan, Raghav Magazine, Shubham Kushwaha, Mérouane Debbah, Deepak Gupta (2024). Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward. IJCAI. https://arxiv.org/pdf/2402.01799",
        "https://arxiv.org/pdf/2402.01799",
    ),
    # --- Carbon footprint / lifecycle analysis ---
    (
        "gupta2022",
        "paper",
        "Chasing Carbon: The Elusive Environmental Footprint of Computing",
        "2022",
        "Udit Gupta, Young Geun Kim, Sylvia Lee, Jordan Tse, Hsien-Hsin S. Lee, Gu-Yeon Wei, David Brooks, Carole-Jean Wu (2022). Chasing Carbon: The Elusive Environmental Footprint of Computing. HPCA '22. https://arxiv.org/pdf/2011.02839",
        "https://arxiv.org/pdf/2011.02839",
    ),
    (
        "faiz2024",
        "paper",
        "LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models",
        "2024",
        "Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, Lei Jiang (2024). LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models. ICLR '24. https://arxiv.org/pdf/2309.14393",
        "https://arxiv.org/pdf/2309.14393",
    ),
    (
        "lannelongue2021",
        "paper",
        "Green Algorithms: Quantifying the Carbon Footprint of Computation",
        "2021",
        "Loic Lannelongue, Jason Grealey, Michael Inouye (2021). Green Algorithms: Quantifying the Carbon Footprint of Computation. Advanced Science. https://arxiv.org/pdf/2007.07610",
        "https://arxiv.org/pdf/2007.07610",
    ),
    # --- Recent 2025-2026 papers ---
    (
        "bommasani2024",
        "paper",
        "The Foundation Model Transparency Index v1.1",
        "2024",
        "Rishi Bommasani, Kevin Klyman, Shayne Longpre, Betty Xiong, Sayash Kapoor, Nestor Maslej, Arvind Narayanan, Percy Liang (2024). The Foundation Model Transparency Index v1.1. arXiv. https://arxiv.org/pdf/2407.12929",
        "https://arxiv.org/pdf/2407.12929",
    ),
    (
        "bannour2024",
        "paper",
        "A Systematic Review of Green AI",
        "2024",
        "Nada Bannour, Sahar Ghannay, Aurelien Nedelec, Anne Vilnat (2024). A Systematic Review of Green AI. Artificial Intelligence Review. https://arxiv.org/pdf/2301.11047",
        "https://arxiv.org/pdf/2301.11047",
    ),
    (
        "tomlinson2024",
        "paper",
        "The Carbon Emissions of Writing and Illustrating Are Lower for AI than for Humans",
        "2024",
        "Bill Tomlinson, Rebecca W. Black, Donald J. Patterson, Andrew W. Torrance (2024). The Carbon Emissions of Writing and Illustrating Are Lower for AI than for Humans. Scientific Reports. https://arxiv.org/pdf/2303.06219",
        "https://arxiv.org/pdf/2303.06219",
    ),
    # --- Additional water-focused papers ---
    (
        "george2023",
        "paper",
        "Measuring the Environmental Impacts of Artificial Intelligence Compute and Applications",
        "2023",
        "Sasha Luccioni, Alex Hernandez-Garcia, Jesse Dodge (2023). Measuring the Environmental Impacts of Artificial Intelligence Compute and Applications. arXiv. https://arxiv.org/pdf/2211.02001",
        "https://arxiv.org/pdf/2211.02001",
    ),
]


def load_existing_ids() -> set[str]:
    """Load existing document IDs from metadata.csv."""
    ids = set()
    if METADATA_CSV.exists():
        with open(METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                ids.add(row["id"].strip())
    return ids


def main():
    apply = "--apply" in sys.argv
    existing = load_existing_ids()

    # Deduplicate: skip papers already in corpus, and skip duplicate IDs/URLs
    to_add = []
    seen_ids = set()
    seen_urls = set()
    skipped = []

    for paper in NEW_PAPERS:
        pid, ptype, title, year, citation, url = paper
        if pid in existing:
            skipped.append((pid, "already in corpus"))
            continue
        if pid in seen_ids:
            skipped.append((pid, "duplicate ID in this batch"))
            continue
        if url in seen_urls:
            skipped.append((pid, f"duplicate URL: {url}"))
            continue
        seen_ids.add(pid)
        seen_urls.add(url)
        to_add.append(paper)

    print(f"\nExisting papers: {len(existing)}")
    print(f"New papers to add: {len(to_add)}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for pid, reason in skipped:
            print(f"  - {pid}: {reason}")

    print()
    for pid, ptype, title, year, citation, url in to_add:
        print(f"  [{year}] {pid}: {title[:70]}...")

    if not to_add:
        print("\nNothing to add.")
        return

    if not apply:
        print(f"\nDry run — pass --apply to actually write {len(to_add)} papers to {METADATA_CSV}")
        return

    # Append to CSV
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for paper in to_add:
            writer.writerow(paper)

    print(f"\nAppended {len(to_add)} papers to {METADATA_CSV}")
    print(f"Total papers: {len(existing) + len(to_add)}")
    print(
        "\nNext steps:\n"
        "  1. cd vendor/KohakuRAG\n"
        "  2. kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n"
        "  3. Copy the new wattbot_jinav4.db to your PPVC\n"
    )


if __name__ == "__main__":
    main()
