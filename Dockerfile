# Same base image as the Codabench evaluation container.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# The submission zip already contains everything needed for inference
# (run.py + metadata.yaml + assets/{student_model.pt, train_prior.npy}).
# Pillow ships with the base image; nothing else is required at inference time.
WORKDIR /workspace

COPY submission_zip/task2_v6_distill.zip /workspace/task2.zip
COPY run.py /workspace/reference_run.py
COPY validate_zip.py /workspace/validate_zip.py

# Optional: local val-split verification (organizers provide the data).
# Mount the UPAR data dir at /data, i.e.
#   docker run --gpus all -v /path/to/upar-data:/data upar2027-track2
# The runner below evaluates the EXACT submission zip through the same
# code path as the Codabench ingestion program.
RUN printf '#!/bin/bash\n\
set -e\n\
mkdir -p /workspace/ingest\n\
cd /workspace/ingest\n\
unzip -o /workspace/task2.zip\n\
python - <<EOF\n\
import importlib.util\n\
spec = importlib.util.spec_from_file_location("run", "/workspace/ingest/run.py")\n\
run = importlib.util.module_from_spec(spec)\n\
spec.loader.exec_module(run)\n\
run.load_model()\n\
print("load_model OK; model loaded:", run._MODEL is not None)\n\
EOF\n\
if [ -d /data ] && [ -f /data/annotations/task2/val/ids.csv ]; then\n\
  cd /workspace && python validate_zip.py task2 /workspace/task2.zip\n\
else\n\
  echo "no val split under /data; smoke-test only"\n\
fi\n' > /workspace/run.sh && chmod +x /workspace/run.sh

CMD ["/workspace/run.sh"]
