## Repository Structure

```
.
├── literature_review/                                    # Literature review materials
│
├── simulation/                                           # Simulation studies
│   ├── RNN-AGT.ipynb                                     # Core RNN-AGT model: uncensored and censored gap-time settings,
│   │                                                     #  Gehan-loss training, nonlinear/interaction-effect experiments (Python)
│   ├── RNN_AGT_v2.ipynb                                  # Updated RNN-AFT model implementation (Python)
│   ├── Mod_Perform_Sub_sampling_Config_v2_ipynb.ipynb    # RNN-AFT with vectorized Gehan-type loss and sub-sampling-pairs strategy (Python)
│   ├── High_Dim_Cov_ipynb.ipynb                          # Experiments and results for high-dimensional covariate settings (Python)
│   └── SimulationPlots_v1.ipynb                          # Plots summarizing simulation results (Python)
│
└── application/                                          # Application to real recurrent-event data
    ├── Application_v1.ipynb                              # Application to real data, incl. comparison to PWP-GT, PWP-TT, WLW (Python)
    ├── Application_v2.ipynb                              # Updated application notebook (Google Drive-based data loading) (Python)
    ├── Dataset1_Data_Preprocessing.R                     # Preprocessing of the readmission (bladder cancer) dataset (R)
    ├── Dataset1_Classical_Recurrent_Event_Models.R       # Classical recurrent-event models fit to the readmission dataset (R)
    ├── Dataset2_Data_Preprocessing.R                     # Preprocessing of the CGD dataset (R)
    └── Dataset2_Classical_Recurrent_Event_Models.R       # Classical recurrent-event models fit to the CGD dataset (R)
```
