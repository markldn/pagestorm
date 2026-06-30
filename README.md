![PageStorm Research Preview](cover_image.png)

# Pagestorm

### Paper: [`2605.17064`](https://huggingface.co/papers/2605.17064)
### Collection: [`PageStorm Research Preview`](https://huggingface.co/collections/Pageshift-Entertainment/pagestorm-research-preview)
### Twitter / X: [`https://x.com/pageshiftAI`](https://x.com/pageshiftAI)
### Job Openings: [`https://pageshift.ai/hiring`](https://pageshift.ai/hiring)

## Install

```bash
pip install "git+https://github.com/Pageshift-ai/pagestorm.git"
```

## Run

### 14B Full Book

```bash
pagestorm generate \
  --repo-id Pageshift-Entertainment/pagestorm-research-preview-14b-full-book \
  --prompt "Thriller in Zurich"
```

### 24B First Chapter

```bash
pagestorm generate \
  --repo-id Pageshift-Entertainment/pagestorm-research-preview-24b-first-chapter-only \
  --prompt "Thriller in Zurich"
```

## CLI Flags

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--repo-id` | Yes | None | Known Pagestorm Hugging Face model repo ID. |
| `--prompt` | Yes | None | Inline prompt text. |
| `--output-directory` | No | None | Directory for `artifact.txt` and the generated-title book text file. |
| `--tensor-parallel-size` | No | `1` | vLLM tensor parallel size. |
| `--gpu-memory-utilization` | No | `0.96` | vLLM GPU memory utilization. |
| `--max-model-len` | No | Repo default | vLLM maximum model context length. |
| `-h`, `--help` | No | None | Show command help. |
