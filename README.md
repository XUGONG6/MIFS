# MIFS: An adaptive multi-path information fusion self-supervised framework for drug discovery

Supporting Information for the paper "MIFS: An adaptive multi-path information fusion self-supervised framework for drug discovery."

MIFS IS an adaptive multi-path information fusion self-supervised framework that learns molecular representations from large-scale unlabeled data to aid drug discovery. In MIFS, we innovatively design a dedicated molecular graph encoder, Mol-EN, which implements three pathways of information propagation: atom-to-atom, chemical bond-to-atom, and substructure-to-atom. Further, a novel adaptive pre-training strategy based on the molecular scaffolds is devised to pre-train Mol-EN on 11 million unlabeled molecules.

![MIFS](./MIFS.png)




## Dataset
All data used in this paper are uploaded and can be accessed. 

- DTI datasets: Human, C.elegans
- DDI datasets: DrugBank, BIOSNAP
- MPP datasets:  Lipo, ESOL, FreeSolv, BACE, BBBP, Tox21, ToxCast, SIDER, and ClinTox.


## To run the training procedure
- Run conda env create -f environment.yaml to set up the environment.
- Create data in PyTorch format: python create_data.py
- Run python pre_training.py to pre-train the model.
- Run python training_validation.py.





The core code of MIFS will be published when the paper review is complete. 
Please stay tuned for updates!
Thank you.



