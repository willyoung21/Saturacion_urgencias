#!/usr/bin/env bash
set -e

# ───────────────────────────────────────────────
#  Saturacion Urgencias — Setup script
#  Usage: bash setup.sh
#  Requirements: Git Bash (Windows), or any Linux/macOS terminal
# ───────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Urgencias · Panel del Coordinador${NC}"
echo -e "${CYAN}   5 niveles de severidad · 3 horizontes${NC}"
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""

# ── Check UV ────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}UV no encontrado. Instalando...${NC}"
    if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo -e "${YELLOW}En Windows, instala UV desde: https://docs.astral.sh/uv/${NC}"
        echo "Luego ejecuta este script de nuevo."
        exit 1
    fi
fi

echo -e "${GREEN}✓ UV encontrado: $(uv --version)${NC}"

# ── Install dependencies ────────────────────────

echo ""
echo -e "${CYAN}Instalando dependencias con uv sync...${NC}"
uv sync
echo -e "${GREEN}✓ Dependencias instaladas${NC}"

# ── Train models? ───────────────────────────────

echo ""
echo -e "${YELLOW}Los modelos predictivos no estan incluidos en el repositorio${NC}"
echo -e "${YELLOW}(pesan ~50 MB cada uno, muy grandes para GitHub).${NC}"
echo ""
echo -e "${BOLD}¿Deseas entrenar los modelos ahora? (s/N):${NC} "
read -r respuesta
if [[ "$respuesta" == "s" || "$respuesta" == "S" ]]; then
    echo ""
    echo -e "${CYAN}Entrenando 3 modelos Random Forest (+1h, +3h, +6h)...${NC}"
    echo -e "${YELLOW}Esto tomara aproximadamente 2-5 minutos...${NC}"
    uv run python notebooks/retrain_model.py
    echo -e "${GREEN}✓ Modelos entrenados y guardados en models/${NC}"
else
    echo ""
    echo -e "${YELLOW}Puedes entrenarlos mas tarde con:${NC}"
    echo "  uv run python notebooks/retrain_model.py"
fi

# ── Done ────────────────────────────────────────

echo ""
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Instalacion completada${NC}"
echo ""
echo -e "${BOLD}Para iniciar el dashboard:${NC}"
echo ""
echo -e "  ${CYAN}uv run streamlit run src/app.py${NC}"
echo ""
echo -e "${BOLD}O con el entorno activado:${NC}"
echo "  source .venv/bin/activate  (Linux/macOS)"
echo "  .venv\\Scripts\\activate    (Windows)"
echo "  streamlit run src/app.py"
echo ""
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""
