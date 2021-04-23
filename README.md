# Vertical Federated Learning on Credit Assessment Data

#### Capstone Project of Risk Management and Business Intelligence in HKUST

## Research Problem
Currently, there are serveral credit assessment methods applied in industries. However, no credit assessement approach is set up for small businesses due to strict risk management after 2008 global finanacial crisis. Potential of loan market for Micro, Small and Medium Enterprises (MSME) is extremely large as they accounts for 98% businesses in Hong Kong. Moreover, public awareness of data privacy is growing because of recent data missused cases.  

This project aims to create tailored credit assessment model for MSMEs while protecting privacy.

## Methodology
Federated Learning is adopted to protect data privacy. 
Deep Learning approach is used for constructing tailored credit assessment method for MSMEs.

In this section, we mainly focus on Vertical Federated Learning.

## Packages
1. **PySyft** - Federated Learning structure
2. **PyVertical** - Vertical Federated Learning framework
3. **PyTorch** - Deep Learning

2 scenarios are set up to examine the effiency of VFL
Additional 2 msme dataset version are also tested for reference
## Uploaded Files
- **VFL_example.py**: main structure of our experiements
- **VFL_DatasetA_final.ipynb**: standard version
- **VFL_DatasetB_final.ipynb**: add more feature columns
- **VFL_msmeA_final.ipynb**: only contains MSME data, same feature columns with DatasetA
- **VFL_msmeB_final.ipynb**: only contains MSME data, same feature columns with DatasetB
