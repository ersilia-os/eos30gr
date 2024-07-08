# Classification of hERG blockers and nonblockers

This model used a multitask deep neural network (DNN) to predict the probability that a molecule is a hERG blocker. It was trained using 7889 compounds with experimental data available (IC50). The checkpoints of the pretrained model were not available, therefore we re-trained the model using the same method but without mol2vec featuriztion. Molecule featurization was instead done with Morgan fingerprints. Six models were tested, with several thresholds for negative decoys (10, 20, 40, 60, 80 and 100 uM). The authors have implemented the 80uM cut-off for negatives. 

## Identifiers

* EOS model ID: `eos30gr`
* Slug: `deepherg`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Classification`
* Output: `Probability`
* Output Type: `Float`
* Output Shape: `Single`
* Interpretation: Probability of hERG blockade. Actives are defined as IC50<10, inactives are defined as IC50>80

## References

* [Publication](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769)
* [Source Code](https://github.com/ChengF-Lab/deephERG)
* Ersilia contributor: [azycn](https://github.com/azycn)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos30gr)
* [AWS S3](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos30gr.zip)
* [DockerHub](https://hub.docker.com/r/ersiliaos/eos30gr) (AMD64, ARM64)

## Citation

If you use this model, please cite the [original authors](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a None license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!