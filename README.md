> ## This is a reference implementation of the Mind Mappings Framework. Mind Mappings performs a gradient-based mapping space search for hardware accelerators.

---

### To Get started, follow the below steps:

- Install the required packages using `pip install -r requirements.txt`
- Install the Mind Mappings package: `pip install -e .`
- Install PyTorch using instruction from [here](https://pytorch.org/get-started/locally/).
- Install [Timeloop](https://github.com/NVlabs/timeloop). Follow instructions.
- Update `parameters.py`:
  - Point to a temporary path and set `self.SCRATCH`
  - Set `self.TIMELOOP_PATH` to point to timeloop path.
- To test if timeloop and mind mappings setup are fine, run `python3 costModel/timeloop/model_timeloop.py`. This should randomly choose a valid mapping and print its cost.

---

Now, everything is setup. Take some time to explore the mind mappings package. `costModel` directory has `models.py`, which describes the mandatory functions that any cost model should implement. `example/` directory contains a simple example cost model for finding minimum of a quadratic equation. `timeloop/` directory shows the mind mappings and time loop interface. 

Everything related to performing mapping space search is handles with `optimize.py`. As understood from the paper, here are two key phases:
1. **Train a Surrogate**
2. **Use Surrogate for Search**

For convenience, two trained surrogate models are already provided for you (in `gradSearch/saved_models/`): `model_CNN-layer.save` and  `model_MTTKRP.save`. Each of them are specific to the architecture described in the paper and the related algorithm. In case, you want to target a different architecture/algorithm, they need to be re-trained (steps are provided later).

To perform mapping space search, run:
    python3 optimize.py --command search --algorithm CNN-layer --problem 16 512 256 3 3 14 14 --maxsteps 1000

`--algorithm` can be set to CNN-layer or MTTKRP, `--problem` should be set to the problem shape (`N C K R S P Q`/`I J K L`, see paper for description), `--maxsteps` can be set to the maximum number of steps you would like the search to run.

This prints out the best mapping and its predicted cost.

---

In case, you would like to train a different surrogate model, follow the steps shown below:

1. **Generate Surrogate Dataset**: `python3 optimize.py --command datagen --path <PATH> --algorithm <ALG> --costmodel <your new cost model>`
2. **Process the Dataset**: `python3 optimize.py --command dataprocess --path <PATH> --algorithm <ALG> --costmodel <your new cost model>`
3. **Train the surrogate model**: `python3 optimize.py --command train --path <PATH> --algorithm <ALG>`
4. **Mapping Space Search**: `python3 optimize.py --command search --algorithm <ALG> --problem <DIMS> --maxsteps <STEPS>`

---

If you would like to reproduce the results from the paper, you can run:

    python3 optimize.py --command reproduce

---

If this was useful in your research, please cite:

```Kartik  Hegde,  Po-An  Tsai,  Sitao  Huang,  Vikas  Chandra,  Angshuman Parashar, and Christopher W. Fletcher. 2021.
Mind Mappings: EnablingEfficient Algorithm-Accelerator Mapping Space Search.
In Proceedings of the 26th ACM International Conference on Architectural Support for ProgrammingLanguages and Operating Systems (ASPLOS ’21), 
April 19–23, 2021, Virtual, MI,USA.ACM, New York, NY, USA```
