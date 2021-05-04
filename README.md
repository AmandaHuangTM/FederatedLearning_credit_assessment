# Federated Learning on Credit Assessment Data

#### Capstone Project of Risk Management and Business Intelligence in HKUST

## Research Problem
Currently, there are serveral credit assessment methods applied in industries. However, no credit assessement approach is set up for small businesses due to strict risk management after 2008 global finanacial crisis. Potential of loan market for Micro, Small and Medium Enterprises (MSME) is extremely large as they accounts for 98% businesses in Hong Kong. Moreover, public awareness of data privacy is growing because of recent data missused cases.  

This project aims to create tailored credit assessment model for MSMEs while protecting privacy.

## Methodology
Federated Learning is adopted to protect data privacy.  
Deep Learning approach is used for constructing tailored credit assessment method for MSMEs.


## Packages
1. **PySyft** - Federated Learning structure, used in HFL and VFL
2. **PyVertical** - Vertical Federated Learning framework, used in VFL
3. **PyTorch** - Deep Learning

Majorly, 2 scenarios are set up to examine the effiency of VFL. They are based on Dataset_A and Dataset_B respectively.
Additional msme dataset version are also tested for investigating the application on MSMEs.  

## Uploaded Files
- **Centralized_example.py**: main structure of centralized model
- **HFL_example.py**: main structure of Horizontal Federated Learning model
- **VFL_example.py**: main structure of Vertical Federated Learning model
### Folders
- **C**: centralized model files on each dataset with output (100 epoches)
  - **200**: centralized model with 200 epoches
- **HFL**: HFL model files on each dataset with output (100 epoches)
  - **200**: HFL model with 200 epoches
- **VFL**: VFL model files on each dataset with output (100 epoches), and one additional PySyft 03 version file
  - **200**: VFL model with 200 epoches
