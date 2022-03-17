To run experiments, run the following files in sequence.

- run `function.generate_gp` to produce a GP function and save the function values at grid points as `function_name.csv`. An example of generating a 2D function `f1` on 100 grid points over [-5, 5]^2 is given below.
```
generate_gp(5, 2, 5, 0.5, 'f1')
```
- run `main.py` to execute `BPE`, `B-BPE`, `B-Fixed-BPE`, and `GP-UCB`.
```
python main.py [function_name] [beta] [reset_gp]
```
  - `function_name` is the name of a generated function (e.g. `f1`).
  - The supported choices of beta are ``[2, 6, log]``, where `log` represents `beta(i) = 3 * log(2i)`. Please edit the dictionary `BETA` on the top of `main.py` to add more choices of beta.
  - `reset_gp` (0 or 1) represents whether to reset GP posterior at the start of each batch.
