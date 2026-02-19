# mvs-graphcuts

Reference implementation of [Multi-View Stereo Using Graph Cuts-Based Depth Refinement](https://ieeexplore.ieee.org/abstract/document/9866789) published in *IEEE Signal Processing Letters* (2022).

## Paper Summary

The paper presents a depth map-based multi-view stereo (MVS) pipeline designed to improve reconstruction quality in **homogeneous / weakly textured regions**, where most MVS methods struggle due to insufficient local photometric evidence.

Instead of refining depth by processing only subsets of pixels (patch-based or iterative local refinement), the proposed method performs **global refinement of all pixels in an image simultaneously** by converting depth refinement into a **graph cut optimization problem**.

The pipeline consists of:

* **Coarse depth estimation** at reduced resolution using a robust photo-consistency objective (ZNCC patch matching + best-κ aggregation), optimized per-pixel using a metaheuristic (e.g., Particle Swarm Optimization).
* **Cross-view consistency filtering** and **cross-view depth completion** to remove inconsistent depths and fill missing values.
* **Graph cuts-based depth refinement**, where depth optimization is formulated as an **s–t min-cut problem** on a **3D grid graph** (rows × cols × depth levels).

  * The graph uses **offset vertices** to align depth hypotheses around each pixel’s initial depth, enabling efficient global refinement without increasing graph size.
  * The energy includes a **data term** (photo-consistency) and a **smoothness term** (depth discontinuity), weighted by an **image-gradient-based adaptive regularizer** to preserve depth discontinuities near edges while enforcing smoothness in flat regions.
* Refined depth maps are fused into a dense point cloud and converted into a watertight mesh using **Poisson surface reconstruction**.

### Results

The method is evaluated on **Middlebury**, **EPFL**, and **DTU** multi-view datasets, demonstrating strong completeness and competitive accuracy, with notable improvements in low-texture surfaces, occluded regions, and challenging illumination.

## Quick start

- Install dependencies:

```
pip install -r requirements.txt
```

## Citation

```
@article{nair2022multi,
  title={Multi-View Stereo Using Graph Cuts-Based Depth Refinement},
  author={Nair, Nirmal S and Nair, Madhu S},
  journal={IEEE Signal Processing Letters},
  volume={29},
  pages={1903--1907},
  year={2022},
  publisher={IEEE}
}
```
