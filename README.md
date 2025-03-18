# iEEG2NWB
Python code to convert human iEEG data to NWB format via command line or GUI. 
Documentation in progress.

## Installation

Instructions on how to install the project.

```python
cd /users/Documents/iEEG2NWB
conda env create -f env.yaml
```

## Usage

Instructions and examples of how to use the project.

## Contributing

Guidelines for contributing to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was done by Noah Markowitz and Stephan Bickel (PI) of the Human Brain Mapping Laboratory at The Feinstein Institutes for Medical Research, Northwell Health.

## Funding

This project was funded by:
- **Kavli Foundation** Seed Grant (Stephan Bickel).
- **R01DC019979 NIMH** (MPI Stephan Bickel).

We are grateful for their support!


# TODO

* Finish `read_ielvis()` full

* Finish new `IEEG2NWB` class

* Complete unit tests

* Create sphinx docs

* `pial_to_inflated()` doesn't need labels arg

* `sub_to_fsaverage()` doesn't need labels arg