# bioemij: BioEM-In Jax

## Installation

We recommend installing this code on a new virtual environment. After activating your environment, install jax for your appropiate CUDA version; for example for CUDA 12, run:

```bash
pip install jax[cuda12]
```
You can find your CUDA version by running`nvida-smi`. However, this code has only been tested with CUDA12. Support for newer cuda version depends on their compatibility with `JAX`.

Once `JAX` is installed, install our library by cloning the repository, and installing the code locally

```bash
git clone git@github.com:miro-astore/bioemij.git
cd bioemij
pip install -e .
```

## Testing
to output timestamps in pytest:
```bash
>  pytest tests  --log-cli-level=INFO   --log-cli-format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"  --log-cli-date-format="%Y-%m-%d %H:%M:%S"
```
This is not quite the bioEM algorithm because it uses a heirarchical pose search rather than numerical marginalisation. For references see 

https://www.sciencedirect.com/science/article/pii/S0010465516302855#s000090

https://pubmed.ncbi.nlm.nih.gov/24161733/

https://www.biorxiv.org/content/10.64898/2025.12.23.696001v1
