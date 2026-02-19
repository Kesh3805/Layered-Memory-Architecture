FROM python:3.12.2-slim-bookworm

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Use a fixed Debian mirror to avoid Hash Sum mismatch
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && printf 'Acquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\nAcquire::BrokenProxy "true";\nAcquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99fixbadproxy \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt .

# Install CPU-only PyTorch first (good move 👍)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
