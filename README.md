# BoltzGen on Amazon SageMaker

Scalable protein design pipeline using [BoltzGen](https://github.com/HannesStark/boltzgen) on AWS SageMaker. This fork enables **large-scale batch processing** with multi-GPU parallelization.

## Key Features

- **Scalable GPU instances** - ml.g4dn, ml.g5, ml.g6e series
- **Batch processing** - Process 100-10,000+ samples efficiently
- **Multi-GPU parallelization** - Up to 8 GPUs per instance
- **Multi-instance scaling** - Add more instances for faster completion
- **Cost-effective** - ~$2.70 per sample with 100 designs

## Quick Start

### Single Design Job

```bash
python sagemaker/run_processing_job.py \
  --design-spec example/hard_targets/1g13prot.yaml \
  --s3-bucket your-bucket \
  --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```

### Batch Processing (100+ samples)

```bash
cd sagemaker/pipeline/examples

# 1. Setup AWS credentials
cp .env.example .env
# Edit .env with your AWS settings

# 2. Prepare example data
mkdir -p prot_only_specs
cp ../../../example/hard_targets/*prot.yaml prot_only_specs/
cp ../../../example/hard_targets/*.cif prot_only_specs/

# 3. Update config with your .env values and run
python run_single_instance_batch.py --config prot_only_config.yaml
```

## Scaling: More Instances = Faster Processing

Each GPU processes one sample at a time. Adding more instances directly reduces total processing time while keeping costs similar.

### Example: 1000 Samples

| Configuration | Total GPUs | Time | Cost |
|--------------|------------|------|------|
| 1x ml.g5.12xlarge | 4 | 375h (15.6 days) | $2,659 |
| 5x ml.g5.12xlarge | 20 | 75h (3.1 days) | $2,659 |
| 10x ml.g5.12xlarge | 40 | 38h (1.6 days) | $2,695 |

**Key insight**: 10x instances = 10x faster at the same cost.

### Instance Pricing (US East - N. Virginia)

| Instance Type | GPUs | Price/hour | Parallel Samples |
|--------------|------|------------|------------------|
| ml.g4dn.xlarge | 1 | $0.74 | 1 |
| ml.g5.xlarge | 1 | $1.41 | 1 |
| ml.g5.12xlarge | 4 | $7.09 | 4 |
| ml.g5.48xlarge | 8 | $20.36 | 8 |

*Source: [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)*

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](sagemaker/QUICKSTART.md) | Get started in minutes |
| [SageMaker Guide](sagemaker/README.md) | Single job setup and usage |
| [Batch Processing](sagemaker/pipeline/README.md) | Large-scale batch processing |
| [Batch Examples](sagemaker/pipeline/examples/README.md) | Scripts and configurations |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BoltzGen on SageMaker                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Design     │    │   Inverse    │    │   Folding    │      │
│  │   Specs      │───▶│   Folding    │───▶│  Validation  │      │
│  │   (S3)       │    │   (GPU)      │    │   (GPU)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                    │              │
│                             ▼                    ▼              │
│                      ┌──────────────┐    ┌──────────────┐      │
│                      │   Analysis   │───▶│   Results    │      │
│                      │   & Ranking  │    │   (S3)       │      │
│                      └──────────────┘    └──────────────┘      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Execution Modes:                                               │
│  • Single Instance: 1x ml.g5.12xlarge (4 GPUs parallel)        │
│  • Multi Instance:  Nx instances for massive parallelization   │
└─────────────────────────────────────────────────────────────────┘
```

---

# BoltzGen (Original)

> This is a fork of [BoltzGen](https://github.com/HannesStark/boltzgen) by Hannes Stark et al.

<div align="center">
  <img src="assets/boltzgen.png" alt="BoltzGen logo" width="60%">

[Paper](https://hannes-stark.com/assets/boltzgen.pdf) |
[Slack](https://boltz.bio/join-slack)

![alt text](assets/cover.png)
</div>

## Installation

```bash
pip install boltzgen
```

<details>
<summary>Detailed installation instructions</summary>

### 1. Install Miniconda

* **Windows:** https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation
* **macOS / Linux:** https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation

### 2. Create environment

```bash
conda create -n bg python=3.12
conda activate bg
pip install boltzgen
```

For development:
```bash
pip install -e .
```
</details>

<details>
<summary>Docker instructions</summary>

```bash
docker build -t boltzgen .

mkdir -p workdir cache
docker run --rm --gpus all \
  -v "$(realpath workdir)":/workdir \
  -v "$(realpath cache)":/cache \
  -v "$(realpath example)":/example \
  boltzgen run /example/vanilla_protein/1g13prot.yaml \
  --output /workdir/test \
  --protocol protein-anything \
  --num_designs 2
```
</details>

## Running BoltzGen

![alt text](assets/fig1.png)

```bash
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2
```

**Step-by-step:**
1. Create your `.yaml` design specification (see `example/` directory)
2. Check with `boltzgen check your_spec.yaml`
3. Run `boltzgen run your_spec.yaml --output ...`
4. Results in `--output` directory

## Protocols

| Protocol | Use Case |
|----------|----------|
| protein-anything | Design proteins to bind proteins or peptides |
| peptide-anything | Design cyclic peptides to bind proteins |
| protein-small_molecule | Design proteins to bind small molecules |
| nanobody-anything | Design nanobodies (single-domain antibodies) |

## Design Specification

Example `.yaml` file:
```yaml
entities:
  - protein:
      id: B
      sequence: 80..140  # Designed protein with 80-140 residues

  - file:
      path: 6m1u.cif
      include:
        - chain:
            id: A
```

See [example/README.md](example/README.md) for detailed documentation.

## Command Reference

```bash
# Run design pipeline
boltzgen run design_spec.yaml --output ./results --protocol protein-anything --num_designs 100

# Check design specification
boltzgen check design_spec.yaml

# Download models
boltzgen download all

# Configure without running
boltzgen configure design_spec.yaml --output ./config
```

## Citation

```bibtex
@misc{stark2025boltzgen,
  title        = {BoltzGen: Toward Universal Binder Design},
  author       = {Hannes Stark and Felix Faltings and MinGyu Choi and others},
  year         = {2025},
  howpublished = {\url{https://hannes-stark.com/assets/boltzgen.pdf}}
}
```

For full documentation, see the [original BoltzGen repository](https://github.com/HannesStark/boltzgen).
