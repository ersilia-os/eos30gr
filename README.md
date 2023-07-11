# Classification of hERG blockers and nonblockers

This model used a multitask deep neural network (DNN) to predict the probability that a molecule is a hERG blocker. It was trained using 7889 compounds with experimental data available (% of hERG inhibition). The checkpoints of the pretrained model were not available, therefore we re-trained the model using a simple KerasTuner. Molecule featurization was done with Mol2vec, accordingly to the original model.

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
* Interpretation: Probability of hERG blockade. The training dataset used a threshold of 80% inhibition to define hERG blockers.

## References

* [Publication](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769)
* [Source Code](https://github.com/ChengF-Lab/deephERG)
* Ersilia contributor: [azycn](https://github.com/azycn)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos30gr)

## Citation

If you use this model, please cite the [original authors](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00769) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a None license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!