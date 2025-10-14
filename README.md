# Classification of hERG blockers and nonblockers

This model used a multitask deep neural network (DNN) to predict the probability that a molecule is a hERG blocker. It was trained using 7889 compounds with experimental data available (IC50). The checkpoints of the pretrained model were not available, therefore we re-trained the model using the same method but without mol2vec featuriztion. Molecule featurization was instead done with Morgan fingerprints. Six models were tested, with several thresholds for negative decoys (10, 20, 40, 60, 80 and 100 uM). The authors have implemented the 80uM cut-off for negatives. 

This model was incorporated on 2022-07-14.Last packaged on 2025-10-14.

## Information
### Identifiers
- **Ersilia Identifier:** `eos30gr`
- **Slug:** `deepherg`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `ADMET`
- **Target Organism:** `Homo sapiens`
- **Tags:** `Toxicity`, `hERG`, `Cardiotoxicity`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability of hERG blockade. Actives are defined as IC50<10, inactives are defined as IC50>80

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| activity_80 | float | high | Probability of hERG blockade with actives defined as IC50<10 and inactives as IC50>80 |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos30gr](https://hub.docker.com/r/ersiliaos/eos30gr)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos30gr.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos30gr.zip)

### Resource Consumption
- **Model Size (Mb):** `41`
- **Environment Size (Mb):** `5967`
- **Image Size (Mb):** `5914.25`

**Computational Performance (seconds):**
- 10 inputs: `33.2`
- 100 inputs: `23.01`
- 10000 inputs: `262.26`

### References
- **Source Code**: [https://github.com/ChengF-Lab/deephERG](https://github.com/ChengF-Lab/deephERG)
- **Publication**: [https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2019`
- **Ersilia Contributor:** [azycn](https://github.com/azycn)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [None](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos30gr
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos30gr
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
