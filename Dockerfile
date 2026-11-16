# Inkling-2 2-GPU test image: the ink10 rc2 image + the current inkling2 model code (TP6 uneven-KV layout, segment-table
# kernels, drafter on its own GPU, DSpark K 1..7, W4A4 prefill switch) + the test set in /opt/inkling-tests.
FROM btbtyler09/vllm-cuda@sha256:00b0f8f9fbac3c4292621247612a2e4a9382db54396f568d93599042164a0580
USER 0
COPY ovl/inkling2/ /opt/rt/lib/python3.12/site-packages/vllm/models/inkling2/
COPY ovl/sconv_swa_attn.py /opt/rt/lib/python3.12/site-packages/vllm/models/inkling/nvidia/sconv_swa_attn.py
COPY tests/ /opt/inkling-tests/
RUN find /opt/rt/lib/python3.12/site-packages/vllm/models/inkling2 /opt/rt/lib/python3.12/site-packages/vllm/models/inkling \
        -name __pycache__ -prune -exec rm -rf {} + \
 && chmod 0755 /opt/inkling-tests/run_all && chmod -R a+rX /opt/inkling-tests \
 && /opt/rt/bin/python -c "import vllm.models.inkling2.speculator as s, vllm.models.inkling2.attention as a; \
assert hasattr(s.Inkling2DSparkSpeculator, '_load_remote_draft') and hasattr(a, 'uneven_kv_layout'); print('inkling2 code ok')"
USER 2000
WORKDIR /work
ENTRYPOINT ["/opt/inkling-tests/run_all"]
