# Pin the multi-arch index of python:3.11-slim, the same minor as CI, so the tag cannot move.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

# Keep bytecode and pip caches out of the image, leave logs unbuffered, and point matplotlib at a writable config dir.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# The build context is the repo root.
WORKDIR /app

# Copy the lock before the code so a source edit does not reinstall dependencies.
COPY requirements-docker.lock .

# Install every pinned distribution. Skip -e lines: those local packages are copied below.
RUN grep -v '^-e ' requirements-docker.lock > /tmp/requirements-docker.lock \
 && pip install -r /tmp/requirements-docker.lock

# The case imports budget-rule and reads data/ and results/ from the editable tree.
COPY budget-rule/ budget-rule/

# geo-holdout-case is the only demo this image runs.
COPY geo-holdout-case/ geo-holdout-case/

# --no-deps keeps the pins above. -e is required: run_case.py resolves data/ via Path(__file__).parents[2].
RUN pip install --no-deps -e ./budget-rule -e ./geo-holdout-case

# uid 1000 is the user the image is checked for, and it is not root.
RUN useradd --uid 1000 --no-log-init app

# run_case writes results/, so that directory is owned by the non-root user.
RUN chown app /app/geo-holdout-case/results

# Every later command, including CMD, runs as that user.
USER app

# pytest and the default module both start in the case directory.
WORKDIR /app/geo-holdout-case

# Default command reruns the geo case.
CMD ["python", "-m", "geo_holdout.run_case"]
