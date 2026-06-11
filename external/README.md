# External Simulator Sources

This repository does not vendor third-party simulator repositories. Clone
the upstream sources into the paths below before running the full reproduction
workflows.

```bash
mkdir -p external

git clone https://github.com/effereds/rlad-core-simulator external/rlad-core-simulator
git -C external/rlad-core-simulator checkout d6a4ff136907eb1bd9e8b4151a9162231ce0ee6a

git clone https://github.com/hongzimao/decima-sim external/decima-sim
git -C external/decima-sim checkout c010dd74ff4b7566bd0ac989c90a32cfbc630d84
```

DeepRM author-source alignment was checked against the public DeepRM reference
implementation during the study. The local adapter code in `src/cisose_deeprm`
contains the source-aligned configuration used for the released artifacts.

The external checkouts are ignored by git to avoid publishing upstream source
snapshots, nested `.git` directories, and local build products.
