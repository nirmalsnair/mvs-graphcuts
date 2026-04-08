# mvs-graphcuts

Reference implementation of the paper [Multi-View Stereo Using Graph Cuts-Based Depth Refinement](https://ieeexplore.ieee.org/abstract/document/9866789) published in *IEEE Signal Processing Letters* (2022).

<img src="https://github.com/user-attachments/assets/8f29b18c-affd-4974-8a08-334e2eaac7cb"  width="600px">

**Interactive 3D models:**  
[![TempleRing](https://img.shields.io/badge/Sketchfab-TempleRing-1CAAD9?logo=sketchfab&logoColor=white)](https://sketchfab.com/3d-models/middlebury-templering-ieee-spl-2022-ed66481b8f3f4b2d8ca991731ac3e4bb) [![DinoRing](https://img.shields.io/badge/Sketchfab-DinoRing-1CAAD9?logo=sketchfab&logoColor=white)](https://sketchfab.com/3d-models/middlebury-dinoring-ieee-spl-2022-7fc184241cd14a46bfd9a9d4464827f0)

## Paper Summary

The paper presents a depth map-based multi-view stereo (MVS) pipeline designed to improve reconstruction quality in **homogeneous / weakly textured regions**, where most MVS methods struggle due to insufficient local photometric evidence.

Instead of refining depth by processing only subsets of pixels (patch-based or iterative local refinement), the proposed method performs **global refinement of all pixels in an image simultaneously** by converting depth refinement into a **graph cut optimization problem**.

![pipeline](https://github.com/user-attachments/assets/aa3f37f4-613a-4981-a9c7-25ad2be7145b)

The pipeline consists of:

* **Coarse depth estimation** at reduced resolution using a robust photo-consistency objective (ZNCC patch matching + best-κ aggregation), optimized per-pixel using a metaheuristic (e.g., Particle Swarm Optimization).
* **Cross-view consistency filtering** and **cross-view depth completion** to remove inconsistent depths and fill missing values.
* **Graph cuts-based depth refinement**, where depth optimization is formulated as an **s–t min-cut problem** on a **3D grid graph** (rows × cols × depth levels).

  * The graph uses **offset vertices** to align depth hypotheses around each pixel’s initial depth, enabling efficient global refinement without increasing graph size.
  * The energy includes a **data term** (photo-consistency) and a **smoothness term** (depth discontinuity), weighted by an **image-gradient-based adaptive regularizer** to preserve depth discontinuities near edges while enforcing smoothness in flat regions.
* Refined depth maps are fused into a dense point cloud and converted into a watertight mesh using **Poisson surface reconstruction**.

### Results

The method is evaluated on **Middlebury**, **EPFL**, and **DTU** multi-view datasets, demonstrating strong completeness and competitive accuracy, with notable improvements in low-texture surfaces, occluded regions, and challenging illumination.

<img src="https://github.com/user-attachments/assets/8a141b9f-8d67-410a-b465-9cb711fae375" width="600px">

###### Figure: Reconstruction results on the [DTU Skull](https://roboimagedata.compute.dtu.dk/) dataset. (Left) input images from different viewpoints, (middle) refined depth maps obtained after graph cuts optimization, (right) reconstructed untextured 3D model.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nirmalsnair/mvs-graphcuts.git
cd mvs-graphcuts
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Download a calibrated multi-view dataset such as [Middlebury MVS](https://vision.middlebury.edu/mview/data/) (e.g., *DinoRing* or *TempleRing*):

Place the images and camera parameters in the dataset directory expected by the scripts.

### 3. Generate coarse depth maps

Compute initial depth maps using ZNCC-based patch matching:

```bash
python dmap_zncc.py
```

### 4. Refine depth maps using Graph Cuts

```bash
python dmap_refinement_gcuts.py
```

### 5. Reconstruct the 3D model

The generated scene is compatible with **Multi-View Environment (MVE)**:

[https://github.com/simonfuhrmann/mve](https://github.com/simonfuhrmann/mve)

Use the MVE tools to convert depth maps into a point cloud and run **Poisson Surface Reconstruction** to obtain the final mesh.

## Citation

```bibtex
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
