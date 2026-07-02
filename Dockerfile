# Decision D23: slim base image — full python image is ~1GB, slim ~150MB.
# Alpine is smaller still but musl libc breaks some wheels; slim is the
# safe default. (Interview: "why slim over alpine?")
FROM python:3.12-slim

WORKDIR /app

# Decision D24: copy requirements FIRST, install, THEN copy code.
# Docker caches layers — this way editing app code doesn't re-download
# every package on rebuild. The classic layer-caching pattern.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Keys are injected at runtime (-e flags), NEVER baked into the image.
# A key in an image layer is a leaked key.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
