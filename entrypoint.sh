#!/bin/sh
# Seed the persistent volume (/app/data) with pre-trained model caches
# that were baked into the Docker image at /app/data_seed.
# Only copies files that are missing in the volume — never overwrites
# user-generated data.

SEED_DIR="/app/data_seed"
DATA_DIR="/app/data"

if [ -d "$SEED_DIR" ]; then
    mkdir -p "$DATA_DIR/cache"
    for f in "$SEED_DIR/cache"/*; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        if [ ! -f "$DATA_DIR/cache/$base" ]; then
            echo "Seeding $DATA_DIR/cache/$base from image …"
            cp "$f" "$DATA_DIR/cache/$base"
        fi
    done
    echo "Model seed check complete."
fi

exec "$@"
