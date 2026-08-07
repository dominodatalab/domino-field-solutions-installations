# INT8 checkpoint patches

## tokenizer_config.json

The INT8 checkpoint's own `tokenizer_config.json` (written by whatever
transformers/tokenizers version `llmcompressor`'s environment used when
saving) sets `"tokenizer_class": "TokenizersBackend"` — not a real
registered class in any transformers version checked so far (base image's
4.51.3, or after upgrading to 5.14.1). Both vLLM serving and plain
HF-backend loading (`AutoTokenizer.from_pretrained`) fail outright on it:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not
currently imported.
```

The file in this directory is a corrected copy — confirmed via byte-diff
against the working FP8-dynamic checkpoint's `tokenizer_config.json` that
`tokenizer_class: PreTrainedTokenizer` is the *only* field that differs.
The underlying `tokenizer.json` vocab is unaffected (also confirmed: the
two checkpoints' `tokenizer.json` files differ by only 98 bytes, consistent
with serialization-version noise, not a different tokenizer).

**Do not mutate the canonical checkpoint in S3 with this file.** Overlay it
at deploy/eval time instead, via a ConfigMap that shadows the one file:

```bash
kubectl create configmap int8-tokenizer-fix -n domino-compute \
  --from-file=tokenizer_config.json=triton-repo-reference/models/llama4scout-vllm-int8/patches/tokenizer_config.json
```

```yaml
volumeMounts:
  - name: tokenizer-fix
    mountPath: /triton-repo/weights/llama4scout-vllm-int8/tokenizer_config.json
    subPath: tokenizer_config.json
volumes:
  - name: tokenizer-fix
    configMap:
      name: int8-tokenizer-fix
```
