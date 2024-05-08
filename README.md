# Fair Regression via Post-Processing

A differentially private post-processing algorithm for fair regression.  Supports *statistical parity* under the *attribute-aware* setting.

To reproduce our results, see the notebooks `law.ipynb` and `communities.ipynb`.

**LP solvers.**  Our algorithm involves solving linear programs, and they are set up in our code using the `cvxpy` package.  For large-scale problems, we recommend the Gurobi optimizer for speed.

## Citation

```bibtex
@inproceedings{xian2024DifferentiallyPrivatePost,
  title     = {{Differentially Private Post-Processing for Fair Regression}},
  booktitle = {{Proceedings of the 41st International Conference on Machine Learning}},
  author    = {Xian, Ruicheng and Li, Qiaobo and Kamath, Gautam and Zhao, Han},
  year      = {2024}
}
```
