# MachineLearningPotential
MachineLearningPotential is a repository containing the training datasets, trained Machine-Learning Potentials, and raw Monte Carlo simulation structures for the equiatomic CrCoNi published with our work ``Quantifying chemical short-range order in metallic alloys``

The atom type mapping for the potential and training dataset are: {0: 'Cr', 1: 'Co', 2: 'Ni'}

## Repository contents

- `Potentials/` — the trained MTP potential (with chemical sampling), `MTP_Cao_20220823.mtp`
- `Training_datasets/` — the training dataset used to fit the MTP potential, `Training_Cao_20220823.cfg`
- `simulations/` — raw Monte Carlo simulation structures, provided as tarballs:
  - `DFTMC_structures.tar.gz` — structures from DFT-Monte Carlo simulations
  - `EAM_structures.tar.gz` — structures from an EAM potential
  - `MLIP_no_Chem_Samp_structures.tar.gz` — structures from the MLIP without chemical sampling
  - `MLIP_Chem_Samp_structures.tar.gz` — structures from the MLIP with chemical sampling

## References & Citing
If you use this repository in your work, please cite:

```
@article{sheriff_quantifying_2024,
	title = {Quantifying chemical short-range order in metallic alloys},
	volume = {121},
	url = {https://www.pnas.org/doi/abs/10.1073/pnas.2322962121},
	doi = {10.1073/pnas.2322962121},
	pages = {e2322962121},
	number = {25},
	journaltitle = {Proceedings of the National Academy of Sciences},
	publisher = {Proceedings of the National Academy of Sciences},
	author = {Sheriff, Killian and Cao, Yifan and Smidt, Tess and Freitas, Rodrigo},
	urldate = {2024-07-10},
	date = {2024-06-18},
}
```

and

```
@article{cao_capturing_2025,
  title = {Capturing Short-Range Order in High-Entropy Alloys with Machine Learning Potentials},
  author = {Cao, Yifan and Sheriff, Killian and Freitas, Rodrigo},
  year = 2025,
  month = aug,
  journal = {npj Computational Materials},
  volume = {11},
  number = {1},
  pages = {268},
  issn = {2057-3960},
  doi = {10.1038/s41524-025-01722-2},
  urldate = {2025-08-21},
  copyright = {All rights reserved},
  langid = {english}
}
```