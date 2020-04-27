## Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback

---

### About

This repository accompanies the real-world experiments conducted in the paper "**Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback**" by [Yuta Saito](https://usaito.github.io/), which has been accepted at [_SIGIR2020_](https://sigir.org/sigir2020/) as a full paper.

If you find this code useful in your research then please cite:
```
@inproceedings{saito2020asymmetric,
  title={Asymmetric tri-training for debiasing missing-not-at-random explicit feedback},
  author={Saito, Yuta},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

### Dependencies

- numpy==1.17.2
- pandas==0.25.1
- scikit-learn==0.22.1
- tensorflow==1.15.2
- optuna==0.17.0
- pyyaml==5.1.2

### Running the code

To run the simulation with real-world datasets,

1. download the Coat dataset from [https://www.cs.cornell.edu/~schnabts/mnar/](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/` directory.
2. download the Yahoo! R3 dataset from [https://webscope.sandbox.yahoo.com/catalog.php?datatype=r](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/` directory.

Then, run the following commands in the `./src/` directory:

- for the MF-IPS models **without** *asymmetric tri-training*
```bash
for data in yahoo coat
do
  for model in uniform user item both nb nb_true
  do
    python main.py -d $data -m $model
  done
done
```

- for the MF-IPS models **with** *asymmetric tri-training* (our proposal)
```bash
for data in coat yahoo
do
  for model in uniform-at user-at item-at both-at nb-at nb_true-at
  do
    python main.py -d $data -m $model
  done
done
```
where (uniform, user, item, both, nb, nb_true) correspond to (*uniform propenisty*, *user propensity*, *item propensity*, *user-item propensity*, *NB (uniform)*, *NB (true)*), respectively.

These commands will run simulations with real-world datasets conducted in Section 5.
The tuned hyperparameters for all models can be found in `./hyper_params.yaml`. <br>
(By adding the `-t` option to the above code, you can re-run the hyperparameter tuning procedure by *Optuna*.)

Once the simulations have finished running, the summarized results can be obtained by running the following command in the `./src/` directory:

```bash
python summarize_results -d coat yahoo
```

This creates `./paper_results/`.

