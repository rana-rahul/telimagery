# [WIP] telimagery

**NOTE: This repository is still in draft mode and not ready for public use. Please feel free however, to check out the code and references below as this repository matures.**

Telimagery provides Python tools for processing and analyzing imaging mass cytometry (IMC) data. Some of the functionalities (will) include: 
- parsing raw image data from IMC experiments
- multi-dimensional image processing/manipulation
- spillover compensation
- single-cell segmentation
- extraction of cellular features (signal intensity for biomarkers or cell state)
- downstream single-cell analysis

## IMC
[IMC](https://www.nature.com/articles/nmeth.2869) is an expansion of mass cytometry (MC), a high-throughput technology capable of powerful multiplexing to interrogate heterogenous cell samples. IMC relies on the same method of detection as MC to analyze immunohistochemical and immunocytological samples with sub-cellular resolution. Here's a snapshot of a typical IMC experiment:

![imc_figure](https://user-images.githubusercontent.com/78240386/213191872-0effc502-9002-4036-8762-3e823bbb7b7b.png)
