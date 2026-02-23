#!/usr/bin/env python3
"""Quick smoke test for perturbation + LINCS integration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alin.perturbation import (
    get_perturbation_signature,
    get_perturbation_responders,
    build_perturbation_response_paths,
    score_combination_by_perturbation,
    get_feedback_genes,
    get_direct_effectors,
)

# Test curated fallback (no LINCS data present)
sig = get_perturbation_signature('KRAS')
print(f'KRAS signature: {sig.target} ({sig.source})')
print(f'  responders: {len(sig.all_responders)}')
print(f'  effectors: {len(sig.direct_effectors)}')

r = get_perturbation_responders('EGFR')
print(f'EGFR responders: {len(r)}')

d = get_direct_effectors('BRAF')
print(f'BRAF effectors: {len(d)}')

f = get_feedback_genes('KRAS')
print(f'KRAS feedback genes: {len(f)}')

# Test score_combination
result = score_combination_by_perturbation(['KRAS', 'EGFR'], {'BRAF', 'AKT1', 'ERK1'})
print(f'Combination score: {result["perturbation_score"]}')

# Test build_paths
paths = build_perturbation_response_paths({'BRAF', 'AKT1', 'MAPK1', 'ERK2', 'PIK3CA', 'MTOR'})
print(f'Paths built: {len(paths)}')
for t, g, c in paths[:3]:
    print(f'  {t}: {len(g)} genes, conf={c}')

print('ALL OK')
