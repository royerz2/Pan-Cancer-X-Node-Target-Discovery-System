#!/usr/bin/env bash
# ============================================================================
# ALIN Data Setup Script
# ============================================================================
# Downloads all required data for the Pan-Cancer X-Node Target Discovery
# System.  Run this on a fresh clone to populate depmap_data/, synergy_data/,
# validation_data/, and (optionally) LINCS L1000 data.
#
# Usage:
#   bash setup_data.sh              # Core data only (~1.2 GB)
#   bash setup_data.sh --lincs      # Core + LINCS Tier 1 (~7.3 GB)
#   bash setup_data.sh --lincs-full # Core + LINCS Full (~56 GB)
#
# Requirements: curl, md5sum (or md5 on macOS), unzip
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }

# ─── Platform-specific md5 ──────────────────────────────────────────────────
md5_check() {
    local file="$1" expected="$2"
    local actual
    if command -v md5sum &>/dev/null; then
        actual=$(md5sum "$file" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        actual=$(md5 -q "$file")
    else
        warn "No md5sum/md5 found — skipping checksum for $file"
        return 0
    fi
    if [[ "$actual" != "$expected" ]]; then
        error "Checksum mismatch for $file (expected $expected, got $actual)"
        return 1
    fi
    return 0
}

# ─── Download helper ────────────────────────────────────────────────────────
download() {
    local url="$1" dest="$2" md5="${3:-}"
    if [[ -f "$dest" ]]; then
        if [[ -n "$md5" ]] && md5_check "$dest" "$md5" 2>/dev/null; then
            info "Already exists (checksum OK): $dest"
            return 0
        elif [[ -z "$md5" ]]; then
            info "Already exists: $dest"
            return 0
        fi
    fi
    info "Downloading: $dest"
    mkdir -p "$(dirname "$dest")"
    curl -L --progress-bar -o "$dest" "$url"
    if [[ -n "$md5" ]]; then
        md5_check "$dest" "$md5" || { error "Download corrupted: $dest"; exit 1; }
    fi
}

# ─── Parse flags ────────────────────────────────────────────────────────────
LINCS_TIER=0  # 0=none, 1=tier1, 2=full
for arg in "$@"; do
    case "$arg" in
        --lincs)      LINCS_TIER=1 ;;
        --lincs-full) LINCS_TIER=2 ;;
        --help|-h)
            echo "Usage: bash setup_data.sh [--lincs|--lincs-full]"
            echo "  (no flag)    Core DepMap + Gygi data only (~1.2 GB)"
            echo "  --lincs      + LINCS Tier 1: CRISPR signatures (~7.3 GB)"
            echo "  --lincs-full + LINCS Full: CRISPR + shRNA + compounds (~56 GB)"
            exit 0 ;;
        *)
            error "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ============================================================================
# 1. CORE DepMap DATA
# ============================================================================
info "=== Core DepMap Data ==="
DEPMAP_DIR="depmap_data"
mkdir -p "$DEPMAP_DIR"

# DepMap 24Q2 — CRISPR (Chronos) dependency scores
CRISPR_URL="https://figshare.com/ndownloader/files/43746708"
download "$CRISPR_URL" "$DEPMAP_DIR/CRISPRGeneEffect.csv"

# DepMap 24Q2 — Cell line metadata
MODEL_URL="https://figshare.com/ndownloader/files/43746705"
download "$MODEL_URL" "$DEPMAP_DIR/Model.csv"

# DepMap 24Q2 — Proteomics (CCLE)
PROT_URL="https://figshare.com/ndownloader/files/40449665"
download "$PROT_URL" "$DEPMAP_DIR/protein_quant_current_normalized.csv"

# DepMap 24Q2 — RNA-seq expression (TPM)
RNA_URL="https://figshare.com/ndownloader/files/43746714"
download "$RNA_URL" "$DEPMAP_DIR/OmicsExpressionProteinCodingGenesTPMLogp1.csv"

# CCLE Sample Info
SAMPLE_URL="https://figshare.com/ndownloader/files/43746693"
download "$SAMPLE_URL" "$DEPMAP_DIR/sample_info.csv"

# ─── Gygi Lab Data (Tables S3, S4, S7) ─────────────────────────────────────
info "=== Gygi Lab Proteomics QC Data ==="

# These files are from the Gygi lab supplementary tables.
# If URLs expire, download manually from the paper's supplementary data:
#   Nusinow et al. 2020, Cell — "Quantitative Proteomics of the Cancer Cell
#   Line Encyclopedia"
# Table S3 — Biological Replicates (Normalized)
if [[ ! -f "$DEPMAP_DIR/Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx" ]]; then
    warn "Table S3 (normalized replicates XLSX) — download manually from Gygi 2020 SI"
fi

# Table S3 — Non-normalized biological replicates (CSV)
if [[ ! -f "$DEPMAP_DIR/ccle_biological_replicates_nonnormalized.csv" ]]; then
    warn "Non-normalized replicates CSV — download manually from DepMap portal"
fi

# Table S4 — RNA/Protein Correlation
if [[ ! -f "$DEPMAP_DIR/Table_S4_Protein_RNA_Correlation_and_Enrichments.xlsx" ]]; then
    warn "Table S4 (RNA/protein correlation) — download manually from Gygi 2020 SI"
fi

# Table S7 — Mutation Associations
if [[ ! -f "$DEPMAP_DIR/Table_S7_Mutation_Associations.xlsx" ]]; then
    warn "Table S7 (mutation associations) — download manually from Gygi 2020 SI"
fi

# ============================================================================
# 2. VALIDATION DATA
# ============================================================================
info "=== Validation Data ==="
mkdir -p validation_data drug_sensitivity_data synergy_data

if [[ ! -f "validation_data/gold_standard_triples.csv" ]]; then
    warn "Gold standard triples — generated by gold_standard.py"
fi

# ============================================================================
# 3. LINCS L1000 DATA (optional)
# ============================================================================
if [[ $LINCS_TIER -ge 1 ]]; then
    info "=== LINCS L1000 Data (Tier 1: CRISPR Signatures) ==="
    LINCS_DIR="lincs_data"
    mkdir -p "$LINCS_DIR"

    # LINCS 2020 base URL (clue.io bulk download)
    # NOTE: These URLs require a CLUE account. If they fail, visit:
    #   https://clue.io/data/CMap2020#LINCS2020
    # and download manually to lincs_data/
    LINCS_BASE="https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5"

    # Metadata files (small, always download)
    download "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt" \
        "$LINCS_DIR/geneinfo_beta.txt" "45c725d17ce6c377f1e7de07b821a5f0"

    download "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt" \
        "$LINCS_DIR/cellinfo_beta.txt" "c4686b4bcd2bad8fa64e229932c8d486"

    download "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt" \
        "$LINCS_DIR/siginfo_beta.txt"

    download "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt" \
        "$LINCS_DIR/compoundinfo_beta.txt" "bf8e3a15ad026b47903c98d625195d24"

    # Tier 1: CRISPR knockout/overexpression signatures (6.07 GB)
    download "$LINCS_BASE/level5_beta_trt_xpr_n142901x12328.gctx" \
        "$LINCS_DIR/level5_beta_trt_xpr_n142901x12328.gctx" \
        "c852ca26affaa144f1b042463036702b"

    info "Tier 1 LINCS download complete (~6 GB)"
fi

if [[ $LINCS_TIER -ge 2 ]]; then
    info "=== LINCS L1000 Data (Tier 2+3: Compounds + shRNA) ==="

    # Tier 2: Compound treatment signatures (33.08 GB)
    download "$LINCS_BASE/level5_beta_trt_cp_n720216x12328.gctx" \
        "$LINCS_DIR/level5_beta_trt_cp_n720216x12328.gctx" \
        "9a82806e2aba6ec2a866cba77bd57fda"

    # Tier 3: shRNA knockdown signatures (10.95 GB)
    download "$LINCS_BASE/level5_beta_trt_sh_n238351x12328.gctx" \
        "$LINCS_DIR/level5_beta_trt_sh_n238351x12328.gctx" \
        "16952edbdc39756370a075b25f874029"

    info "Full LINCS download complete (~50 GB)"
fi

# ============================================================================
# 4. API CACHE DIRECTORY
# ============================================================================
mkdir -p api_cache/protein

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
info "=== Setup Complete ==="
info "Core data:     $DEPMAP_DIR/"
[[ $LINCS_TIER -ge 1 ]] && info "LINCS data:    lincs_data/"
echo ""
info "Next steps:"
info "  1. Run tests:     python -m pytest tests/ -q"
info "  2. Run pipeline:  python pan_cancer_xnode.py --all-cancers --triples --output results/"
[[ $LINCS_TIER -ge 1 ]] && info "  3. LINCS index:   python -c 'from alin.lincs import LINCSSignatureDB; LINCSSignatureDB(\"lincs_data\").build_index()'"
