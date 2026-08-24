I approve everything written here. Go ahead with implementation. I reviewed the current `main` branch at commit `84bf13965ed5f17ad176518a9db0d34652b2a5e8`, committed on August 23, 2026 UTC—August 24 in Korea. The latest work is the native Qt analysis console and telemetry validation. The branch is currently unprotected, with required status checks disabled.
My overall view is that your direction is sound, and the repository is already farther along than the proposal implies. The main risk is treating this as a broad “rewrite Python in C++” project. I would instead frame it as establishing three clean products:
1. A native application with no Python dependency.
2. A Python training and research system.
3. A temporary Python–C++ parity bridge used only for migration and validation.
That distinction will let you remove Python from the shipped application without spending months rewriting useful training and experimental tooling.
## 1. Native tests and CI
This should be a high priority, with one important adjustment: use the Makefile as a convenient command interface, not as the underlying build system.
The repository already has a suitable native build foundation. CMake is the authoritative build system, uses C++20 and CTest, and makes pybind optional and disabled by default. The native project is already divided into `soo_core`, `soo_search`, optional model support, optional pybind, and optional Qt.
I would therefore use:
* CMake to define targets, dependencies, platform differences and installation.
* CTest to discover and run native tests.
* `CMakePresets.json` to define reproducible build configurations.
* A root Makefile to expose memorable commands such as `make test-native`.
* GitHub Actions YAML to invoke those commands or the corresponding CMake presets.
“Rewrite CI in C++” should mean that CI compiles and executes the C++ tests. The CI definition itself will still be GitHub Actions YAML.
At present, the CI remains substantially Python-driven. The core job runs pytest across Python 3.11–3.13; the native job builds a pybind extension with Python and then runs `pytest tests/native`; the GUI job installs PySide and runs Python GUI tests. It does not currently make the root CMake/CTest build the primary validation route.
The test imbalance is significant. Native C++ currently has only:
* A small action encoding smoke test.
* A deployment artifact contract test.
The smoke test uses ordinary C++ `assert`, which can disappear when `NDEBUG` is defined, so it should be replaced by explicit failure checks or a real test framework.
Meanwhile, the important native correctness gates remain in Python: rules parity, deterministic and stochastic MCTS parity, batching, scheduling, callbacks, self-play pools and topology. They operate over a committed 489 KB position corpus and compare observable behavior in detail, including action ordering, every successor state, canonical mappings, exact feature values, evaluator request sequences and MCTS tie-breaking.
Those Python parity tests should not be deleted immediately. They are currently proving that the port did not change the game. A safe migration is:
**First**, convert the corpus and expected results into a language-neutral, versioned golden-test format. A C++ test should be able to read a record containing the input state and authoritative expected outputs without invoking Python.
**Second**, write C++ tests covering the same observable contracts. They should not merely call one C++ function and compare it with another C++ function that shares the same implementation.
**Third**, keep a smaller Python–C++ parity CI job for a transition period. This becomes the optional “bridge parity” lane.
**Fourth**, retire each Python parity test only after the corresponding C++ golden test catches the known mutations and historical bugs that the Python test caught.
The first native test groups I would create are:
* `board_test`
* `topology_test`
* `action_codec_test`
* `rules_golden_test`
* `state_transition_golden_test`
* `encoder_golden_test`
* `prior_golden_test`
* `mcts_deterministic_golden_test`
* `mcts_stochastic_test`
* `batcher_test`
* `scheduler_test`
* `selfplay_test`
* `deployment_artifact_test`
The Qt tree already defines native controller and analysis smoke tests through CTest, which should become part of the native CI rather than being overshadowed by the PySide test job.
A suitable Makefile interface would look conceptually like this:
```make
.PHONY: configure build test test-native test-python test-parity package clean
PRESET ?= native-ci
configure:
	cmake --preset $(PRESET)
build: configure
	cmake --build --preset $(PRESET) --parallel
test: test-native
test-native: build
	ctest --preset $(PRESET) --output-on-failure
test-python:
	python -m pytest python/tests
test-parity:
	python -m pytest tests/parity
package:
	cmake --build --preset native-release --target package
clean:
	cmake -E rm -rf build
```
The Makefile should never contain compiler source lists or platform-specific linker settings. Those belong in CMake, where they already largely exist.
A good initial definition of done is:
```bash
make test-native
```
must succeed on a clean checkout with no Python interpreter, no pytest and no pybind installed.
## 2. `TrainAlphaDiamond` as Hugging Face bucket storage
Using a Hugging Face Storage Bucket for mutable checkpoints, replay data, logs and intermediate training data is appropriate. Hugging Face describes buckets as mutable, non-versioned object storage intended for checkpoints and intermediate data; it recommends versioned model or dataset repositories for finished artifacts that need history. `hf sync` is supported, updates or adds files by default, and deletion is an explicit and potentially destructive option. ([Hugging Face][1])
I would not, however, make all of `TrainAlphaDiamond` part of Git history.
Treat it as a local synchronization root contained inside the working tree but ignored by Git. Only documentation, manifest schemas and perhaps small release indexes should be tracked:
```text
TrainAlphaDiamond/
  README.md                    tracked
  manifests/                   tracked
  checkpoints/                ignored
    soo/
    min/
  datasets/                   ignored
    selfplay/
    replay/
    fixtures/
  releases/                   ignored
    soo/
    min/
  logs/                       ignored
  staging/                    ignored
```
Your proposed name and command currently disagree: the folder is called `TrainAlphaDiamond`, while the command syncs `./data`. Pick one canonical path. With the proposed folder name, the command becomes:
```bash
hf sync ./TrainAlphaDiamond hf://buckets/ubraket513/AlphaDiamond
```
I would provide separate Make targets with explicit direction:
```make
data-push-dry-run:
	hf sync ./TrainAlphaDiamond \
	  hf://buckets/ubraket513/AlphaDiamond \
	  --dry-run
data-push:
	hf sync ./TrainAlphaDiamond \
	  hf://buckets/ubraket513/AlphaDiamond
data-pull-dry-run:
	hf sync hf://buckets/ubraket513/AlphaDiamond \
	  ./TrainAlphaDiamond \
	  --dry-run
data-pull:
	hf sync hf://buckets/ubraket513/AlphaDiamond \
	  ./TrainAlphaDiamond
```
There should be no ambiguous `make sync` command, and no ordinary target should use `--delete`. Deletion should require a separately named manual operation, preferably after reviewing a generated plan.
Because the bucket is mutable and non-versioned, do not make paths such as `latest.pt` the canonical identity of a checkpoint. Use immutable paths:
```text
checkpoints/
  soo/
    soo-scratch-20260822/
      step-00044250/
        checkpoint.pt
        manifest.json
  min/
    <run-id>/
      step-<number>/
        checkpoint.pt
        manifest.json
```
A `latest.json` file may point to an immutable checkpoint, but the release process should always consume the immutable path and verify its SHA-256.
Each manifest should record at least:
* Model family: Soo or Min.
* Run ID and training step.
* Source Git commit.
* Network architecture and model version.
* Rules, topology, encoder and action-space versions.
* Checkpoint SHA-256.
* Relevant training dataset or replay identifiers.
* Evaluation suite and recorded metrics.
* Framework/runtime versions.
* Whether optimizer, scheduler and RNG state are included.
* Whether the checkpoint is archival, candidate or promoted.
This solves a larger problem than file storage: it establishes provenance.
## 3. Checkpoint conversion and native-app bundling
For Soo, much of the conversion architecture already exists. `tools/export_soo_deployment.py` converts the Python checkpoint into a deployment artifact containing explicit metadata, a TorchScript graph, raw weight tensors, topology data and a deterministic parity corpus. It records hashes and compatibility versions.
There is also strict metadata validation on both sides, and the native artifact contract deliberately corrupts metadata and tensors to prove that invalid artifacts are rejected.
That is a strong foundation. I would refine the goal in two ways.
First, distinguish a **training checkpoint** from a **deployment artifact**:
* A training checkpoint may contain optimizer state, scheduler state, replay compatibility, RNG state and training bookkeeping.
* A deployment artifact should contain only what the native application needs for inference, plus compatibility metadata and integrity hashes.
The application should never bundle the raw training checkpoint.
Second, convert all checkpoints only for archival validation if that is useful, but bundle only promoted releases. Bundling every historical checkpoint will enlarge the installer, create a permanent compatibility obligation and provide little user value. A practical policy would be:
* Preserve every checkpoint and training dataset in the bucket.
* Convert release candidates during promotion.
* Publish only accepted deployment artifacts.
* Bundle one default Soo model and, later, one default Min model.
* Optionally make other released models downloadable rather than bundled.
“Bundled with the app” should generally mean included beside the executable in the packaged application, not compiled into the executable as a byte array:
```text
AlphaDiamond/
  bin/
    diamond_qt
  models/
    index.json
    soo/
      2.0.0/
        metadata.json
        model.ts
        weights/
    min/
      <version>/
        metadata.json
        ...
  licenses/
```
The application can read `models/index.json`, choose the default model, validate the artifact and verify its hash before loading it. This allows model-only updates without relinking the application and makes it easier to ship different defaults in different releases.
There are two current issues to address before generalizing this to Min:
1. The deployment metadata is strongly hardcoded to Soo model version `2.0.0`, width `128` and six residual blocks. That strictness is good for safety but is not yet a general multi-model artifact schema.
2. The Python network directory has both Min and Soo models, while the native LibTorch/model targets currently refer specifically to `soo_model.cpp` and `soo_evaluator.cpp`. Min therefore needs its own native inference implementation and artifact contract before it can be bundled honestly.
I would introduce artifact format version 3 before adding Min, with a structure such as:
```json
{
  "format_version": 3,
  "model_family": "soo",
  "model_version": "2.0.0",
  "architecture": {
    "type": "directional_residual",
    "width": 128,
    "residual_blocks": 6
  },
  "game_contract": {
    "topology": "diamond73-v1",
    "encoder": "diamond-camp-relative-v1",
    "action_space": "diamond73-srcdst-v1"
  },
  "source": {
    "checkpoint_sha256": "...",
    "training_commit": "...",
    "training_step": 44250
  },
  "runtime_sha256": "..."
}
```
The current exporter writes both TorchScript and raw weights. That is useful during parity development, but the production package should eventually have one canonical representation. Otherwise each model is stored twice. Decide whether the native runtime will primarily consume:
* TorchScript through LibTorch.
* Raw validated tensors through the native model implementation.
* A different inference format such as ONNX.
Keep the second representation only as a diagnostic or conversion fixture if it is still needed.
## 4. Repository cleanup
This is more urgent than it may appear.
The repository currently tracks `build-pybind`, including Ninja state, CMake cache files, generated test files and other build outputs.
It also tracks runtime checkpoint directories, including an approximately 9.1 MB `latest.pt`. The current native CI explicitly verifies the hash of another committed runtime checkpoint, so CI and source storage are coupled.
The `.gitignore` already contains patterns intended to ignore build directories, but ignored patterns do not remove files that are already tracked. It also leaves `/runtime/` commented out.
The first cleanup commit should therefore:
* Remove `build-pybind` from Git tracking.
* Remove training checkpoints and run output from Git tracking.
* Move CI fixtures to an explicit fixture mechanism.
* Add `CMakePresets.json`.
* Add the Makefile façade.
* Establish `TrainAlphaDiamond` and its ignore rules.
* Add a repository hygiene test that rejects generated directories and unexpected large binaries.
* Protect `main` and require the new checks once they are green.
A reasonable ignore policy would be:
```gitignore
/build/
/build-*/
/out/
/artifacts/
/runtime/runs/
/TrainAlphaDiamond/**
!/TrainAlphaDiamond/README.md
!/TrainAlphaDiamond/manifests/
!/TrainAlphaDiamond/manifests/**
*.pt
*.pth
*.ckpt
*.safetensors
```
Use exceptions for any intentionally committed tiny fixture files. Blanket extension rules are useful, but a test fixture should be explicitly documented rather than slipping through accidentally.
For CI fixtures, there are two good choices:
* Keep small, immutable golden fixtures in `tests/golden`.
* Download larger model fixtures from an immutable bucket path, checking SHA-256 before running tests.
The current `.gitattributes` marks your own tests, docs and benchmark folders as “vendored.” That only changes GitHub language reporting; it does not solve repository size or organization.
A clean eventual structure could be:
```text
apps/
  qt/
    cpp/
    qml/
    assets/
native/
  include/
  src/
  tests/
python/
  pyproject.toml
  src/diamond/
  tests/
tests/
  golden/
configs/
  training/
  runtime/
tools/
  training/
  conversion/
  benchmark/
  release/
docs/
  architecture/
  training/
  native/
  performance/
TrainAlphaDiamond/
  README.md
  manifests/
```
I would not do all moves in one enormous commit. First remove generated payloads and establish boundaries. Then move the Qt application, Python package, tools and docs in a few history-preserving commits. Large mixed renames make regressions and review unnecessarily difficult.
If clone size is already a concern, removing the files from the latest tree will not remove them from Git history. A coordinated `git filter-repo` history rewrite may eventually be worthwhile, but that should be a separate, announced operation because it changes commit hashes and requires contributors to re-clone or reset their branches.
## 5. What else should move from Python to C++?
I would use three criteria. Port something when it is:
* On the shipped application’s runtime path.
* A measured performance bottleneck.
* A duplicated behavioral authority that creates correctness risk.
Under those criteria, the priorities are fairly clear.
| Area                                      | Recommendation                                   | Reason                                                                |
| ----------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| Board, moves, rules and state transitions | Make C++ authoritative now                       | Already implemented natively; duplicate Python rules are a drift risk |
| Action encoding and canonical topology    | Make C++ or neutral generated data authoritative | Needed by native app and model compatibility                          |
| Encoder and bootstrap prior               | Native authority now                             | Already ported and heavily parity-tested                              |
| MCTS, batcher, scheduler and self-play    | Native authority now                             | Already implemented; core performance path                            |
| Native application session and agents     | Port next                                        | Removes Python from the shipped runtime completely                    |
| Soo inference                             | Complete and stabilize                           | Required by native release packaging                                  |
| Min inference                             | Port before Min is bundled                       | Current native model path is Soo-specific                             |
| Arena game execution                      | Reuse C++ core                                   | Many games benefit from the native engine                             |
| Arena statistics and reports              | Keep Python                                      | Flexible, low runtime value from a C++ rewrite                        |
| Replay parsing and ingestion              | Profile first                                    | Port only if measured as a training bottleneck                        |
| Dataset transformations                   | Profile first                                    | Often suitable for C++, but only when throughput justifies it         |
| Training loop and autograd                | Keep Python/PyTorch                              | Rewriting provides little strategic benefit                           |
| Optimizer and checkpoint surgery          | Keep Python                                      | Experimental tooling changes frequently                               |
| Experiment orchestration and sweeps       | Keep Python                                      | Python is the better control-plane language                           |
| Rating, bootstrap intervals and analysis  | Keep Python                                      | Research/statistical code benefits from rapid iteration               |
| Hugging Face synchronization              | Keep external CLI tooling                        | It is not part of the native application                              |
| Checkpoint export                         | Keep as a controlled Python release tool         | Source checkpoint is a PyTorch artifact                               |
The current Python game package still contains board, move, rules, session and state implementations, despite the corresponding native headers and implementations.
The best final architecture is likely:
```text
C++ game/search/model core
          │
          ├── native Qt application
          │
          └── optional pybind module used by Python training
                          │
                          └── Python training/orchestration/research
```
That makes C++ the source of truth while retaining Python where it is most productive. During migration, the Python implementation can remain as an oracle or fallback, but it should not remain a second permanent rules engine.
For topology specifically, avoid manually maintaining the same tables in both languages. Define a neutral topology specification or a deterministic code-generation step that emits both C++ data and Python test fixtures. The key is one source of truth, not simply moving the duplication from one language to another.
## Recommended sequence
I would organize the effort into five milestones.
**Milestone 0: contracts and boundaries**
Document the native application, Python trainer and bridge as separate products. Define model artifact format v3 and the training-data manifest. Freeze the existing golden parity corpus and its generator version.
**Milestone 1: native build and CI**
Add CMake presets, the Makefile façade and a native CI matrix. Port the straightforward unit and contract tests into C++. Keep the Python parity job separate. Add sanitizer coverage on Linux and native Qt smoke tests.
A sensible CI arrangement is:
* `native-core`: Linux, macOS and Windows; no Python.
* `native-sanitizers`: Linux ASan/UBSan.
* `native-qt`: Linux and Windows.
* `native-model`: LibTorch plus immutable deployment fixture.
* `python-training`: training and research pytest suite.
* `bridge-parity`: pybind and cross-language parity.
* `lint`: clang-format/clang-tidy plus ruff.
* `release`: tags or manual dispatch only.
**Milestone 2: storage and cleanup**
Remove tracked builds, checkpoints and raw run output. Establish `TrainAlphaDiamond`, immutable bucket paths and manifests. Change CI to retrieve large fixtures by exact path and hash.
**Milestone 3: release-grade model artifacts**
Generalize the deployment schema. Promote and package the selected Soo model. Add Min native inference and its contract before promoting Min. Produce checksummed application packages that contain deployment artifacts beside the executable.
**Milestone 4: remove duplicate runtime Python**
Make the native core authoritative for game rules, encoding, MCTS and self-play. Convert Python training to use the native binding. Retire Python implementations only after the native golden tests and bridge parity tests prove the transition.
**Milestone 5: measured additional ports**
Profile replay, dataset processing and arena execution. Port only the portions that are materially expensive or correctness-sensitive.
## Bottom line
I would proceed with the goal, but with these two firm changes:
1. **Do not replace CMake with Make.** Add a Makefile as the stable, human-friendly façade over CMake and CTest.
2. **Do not commit the contents of `TrainAlphaDiamond`.** Keep it under the working tree as an ignored bucket synchronization root, with tracked manifests and documentation.
The immediate target should be:
```bash
git clone ...
make test-native
make package
```
working without Python, while:
```bash
make test-python
make test-parity
```
remain separate developer and training checks.
The repository already has a strong native core, careful parity gates and a notably good Soo deployment-artifact contract. The safest path is to preserve those correctness guarantees while changing which implementation is authoritative—not to delete the Python evidence before its native replacement can prove the same things.
[1]: https://huggingface.co/docs/hub/storage-buckets "https://huggingface.co/docs/hub/storage-buckets"
