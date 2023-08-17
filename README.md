# Investigating generalization behavior in neural networks via Hessian analysis

## Relevant posts

- [2023-08-14: Decomposing independent generalizations in neural networks via Hessian analysis](https://www.lesswrong.com/posts/8ms977XZ2uJ4LnwSR/decomposing-independent-generalizations-in-neural-networks)

## Codebase

- `mnist/` contains the code for the MNIST experiments (modified MNIST digit classification task with superimposed random patterns)
- `modular_addition/` contains the code for the modular addition experiments (addition of two numbers embedded as tokens modulo n)
- `results/` contains interesting results and artifacts from experiments
- `diagrams/` contains code and plots used to generate diagrams in the posts
- `misc/` contains code not directly related to the experiments or referenced in the posts, produced during the course of the project

### Installing dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```