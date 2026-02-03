# KohakuRAG_UI setup (uv, GB10/ARM-friendly)

This repo vendors known-good copies of:

- `vendor/KohakuRAG/` (with `umap-learn` removed to avoid `numba -> llvmlite` issues on GB10 / newer Python)
- `vendor/KohakuVault/` (with the `c_char` FFI fix applied for ARM/Rust)

**Result:** you should not need to edit any upstream files during setup.

Python 3.10+ required. Python 3.11 recommended.

---

## 1) Clone the UI repo and check out the branch you want

```bash
git clone https://github.com/matteso1/KohakuRAG_UI.git
cd KohakuRAG_UI
git checkout local
```

Verify the vendored deps exist:

```bash
ls vendor/KohakuRAG vendor/KohakuVault
```

---

## 2) Create and activate a venv with uv

```bash
uv venv --python 3.11
source .venv/bin/activate
python --version
```

Optional sanity check:

```bash
python -c "import sys; print(sys.executable)"
```

---

## 3) Install vendored KohakuVault + KohakuRAG (editable)

Run from the repo root (`KohakuRAG_UI/`), with the venv active.

### GB10 / ARM (recommended order)

```bash
uv pip install -e vendor/KohakuVault
python -c "import kohakuvault; print('kohakuvault ok')"

uv pip install -e vendor/KohakuRAG
python -c "import kohakurag; print('kohakurag ok')"
```

### x86 (also works)

Using the same steps above is recommended for consistency.

---

## 4) Install the UI repo itself (editable)

```bash
uv pip install -e .
```

---

## 5) Install KohakuEngine (if needed)

The PyPI package name is `kohaku-engine`.
- import name: `kohakuengine`
- CLI tool: `kogine`

```bash
uv pip install kohaku-engine
python -c "import kohakuengine; print('kohakuengine ok')"
kogine --help
```

---

## 6) Quick verification

```bash
python -c "import kohakuvault, kohakurag; print('imports ok')"
```

---

## Notes

### About UMAP
`umap-learn` is intentionally not included in the vendored KohakuRAG dependency set to avoid
`llvmlite/numba` build issues on some systems.

If you want UMAP later, install it manually (may fail on GB10 depending on Python/arch):

```bash
uv pip install umap-learn
```

### Updating vendored dependencies
If you update `vendor/KohakuRAG` or `vendor/KohakuVault` and want your environment to pick up changes:

```bash
uv pip install -e vendor/KohakuVault
uv pip install -e vendor/KohakuRAG
```

(Editable installs usually reflect changes immediately, but reinstalling is a quick reset.)
