# NCPU

## Research plan:

PIOTR
RQ: can NCA recovery features be used to train logic processed on conceptual intelligent material (NCA Canvas) with ability to whitstand environmental damages (radiation)
1. Use unmuttable layers to tell gate where is input and where is output -> this should let NCA learn to transfer energy at the distance
2. Use another unmuttable layers to generalize one model to handle 4 gates (AND,OR,XOR,NAND) -> this should allow us to code which gate we want to use
3. Use another unmuttable layer to add clock -> training NCA to transfer in N-clock units,
4. Add continuous guass noise (added in every step of NCA) to mimic radiation,
5. Have full model which can withstand different levels of radiation and can use all 4 gates,
6. Combine model into large board (can be section activated) which executes some simple Digital Circuit

## How To

```
python3 -m src.learn_pattern
```

## Dev setup

```bash
uv run nbstripout --install --attributes .gitattributes
```

## Stup and project config
- We use uv for project management
- uv setup the project as a proper pytohn package which we can refer to using the module name `ncpu`

## Possible conferences to present to
- (31 January 2026) https://attend.ieee.org/wcci-2026/
- (Friday, April 3rd 2026) https://automataandacri2026.ugent.be/deadlines/
- (26 Jan 2026) https://gecco-2026.sigevo.org/HomePage
- (Possibly March-April 2026) Alife

**Higher bar**
- (Sept 26) https://iclr.cc/Conferences/2026/Dates
- (Jan 28 '26) https://icml.cc/Conferences/2026
- (the conf happens in december) https://neurips.cc/Conferences/2025/Dates

## Literature review:

1. https://arxiv.org/pdf/2505.13058 - A Path to Universal Neural Cellular Automata
2. https://google-research.github.io/self-organising-systems/difflogic-ca/
3. https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch/tree/master
4. MaCE (Mass conserving Dynamics for CA) https://arxiv.org/pdf/2507.12306 
