FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/othyn/immich-declutter" \
      org.opencontainers.image.description="Tags clutter candidates (memes, screenshots, receipts, blurry and low-quality shots, ...) in Immich" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /data && chown app:app /data

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY --chown=app:app *.py ./

USER app

VOLUME /data

CMD ["python", "-u", "declutter.py"]
