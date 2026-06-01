# =============================================================================
# Stage 1: Builder — instala dependencias y entrena modelos
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar UV (gestor de paquetes ultra-rápido)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copiar solo archivos de dependencias (cachea capa si no cambian)
COPY pyproject.toml uv.lock ./

# Instalar dependencias del proyecto (sin dev)
RUN uv sync --frozen --no-dev

# Copiar el resto del código
COPY . .

# Entrenar los 3 modelos Random Forest (+1h, +3h, +6h)
# Esto genera los .pkl en models/ dentro de la imagen
RUN uv run python notebooks/retrain_model.py

# =============================================================================
# Stage 2: Runtime — imagen final limpia y liviana
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copiar Python y UV del builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copiar el entorno virtual completo con dependencias + modelos entrenados
COPY --from=builder /build/.venv ./.venv
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models
COPY --from=builder /build/data ./data
COPY --from=builder /build/notebooks ./notebooks
COPY --from=builder /build/pyproject.toml ./pyproject.toml

# Puerto que expone Streamlit
EXPOSE 8501

# Healthcheck: Streamlit responde en la raíz
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

# Comando por defecto: inicia el dashboard
ENTRYPOINT ["uv", "run", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
