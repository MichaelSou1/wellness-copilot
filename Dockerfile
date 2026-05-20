ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
RUN set -eux; \
    if [ -n "${DEBIAN_MIRROR}" ]; then \
      sed -i \
        -e "s|https://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|https://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|http://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|https://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev libgomp1 curl tini tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    HF_ENDPOINT=https://hf-mirror.com

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "scripts/wechat_ilink_worker.py"]
