**Table A1. Permutation importance of LSTM input channels (exploratory, single seed)**

| label        | channel         | is_pc   |   perm_dRMSE_mean |   perm_dRMSE_sd | signal_over_2sd   | harmful   |   ablation_dRMSE_diagnostic |
|:-------------|:----------------|:--------|------------------:|----------------:|:------------------|:----------|----------------------------:|
| pca_cev_0.75 | PC1             | True    |            0.9500 |          0.1927 | True              | False     |                      0.3781 |
| pca_cev_0.75 | PC2             | True    |            0.3655 |          0.1314 | True              | False     |                      4.7496 |
| pca_cev_0.75 | VNINDEX_history | False   |           18.2298 |          1.4019 | True              | False     |                     37.6955 |
| pca_cev_0.80 | PC1             | True    |            0.4424 |          0.1997 | True              | False     |                     -0.0422 |
| pca_cev_0.80 | PC2             | True    |            0.3809 |          0.1500 | True              | False     |                      7.7007 |
| pca_cev_0.80 | PC3             | True    |            0.2923 |          0.4004 | False             | False     |                     17.2349 |
| pca_cev_0.80 | VNINDEX_history | False   |           12.3047 |          1.0807 | True              | False     |                     32.0704 |
| pca_cev_0.90 | PC1             | True    |            5.7950 |          0.5837 | True              | False     |                      6.3943 |
| pca_cev_0.90 | PC2             | True    |           -0.3471 |          0.0641 | False             | True      |                     -0.1308 |
| pca_cev_0.90 | PC3             | True    |           -0.2445 |          0.0731 | False             | True      |                     -0.0217 |
| pca_cev_0.90 | PC4             | True    |           -0.6265 |          0.0781 | False             | True      |                      6.7290 |
| pca_cev_0.90 | PC5             | True    |           -0.3577 |          0.0445 | False             | True      |                     -0.4284 |
| pca_cev_0.90 | PC6             | True    |           -1.1345 |          0.3555 | False             | True      |                     32.2303 |
| pca_cev_0.90 | VNINDEX_history | False   |            3.6933 |          0.5164 | True              | False     |                      7.3486 |
| pca_cev_0.95 | PC1             | True    |            4.0325 |          0.5118 | True              | False     |                      5.1691 |
| pca_cev_0.95 | PC2             | True    |           -0.6355 |          0.1429 | False             | True      |                     -0.2177 |
| pca_cev_0.95 | PC3             | True    |            0.3006 |          0.1343 | True              | False     |                      3.1172 |
| pca_cev_0.95 | PC4             | True    |           -0.2934 |          0.1181 | False             | True      |                      7.3545 |
| pca_cev_0.95 | PC5             | True    |            0.4486 |          0.1519 | True              | False     |                      1.3225 |
| pca_cev_0.95 | PC6             | True    |           -0.5700 |          0.3001 | False             | True      |                      9.7132 |
| pca_cev_0.95 | PC7             | True    |            4.6319 |          0.8037 | True              | False     |                     11.7171 |
| pca_cev_0.95 | PC8             | True    |           -0.3476 |          0.5272 | False             | True      |                      0.9614 |
| pca_cev_0.95 | PC9             | True    |           -0.2326 |          0.0433 | False             | True      |                      0.7278 |
| pca_cev_0.95 | PC10            | True    |            0.0221 |          0.5605 | False             | False     |                     44.7934 |
| pca_cev_0.95 | PC11            | True    |           -0.1179 |          0.0833 | False             | True      |                     -0.2211 |
| pca_cev_0.95 | VNINDEX_history | False   |            3.6380 |          0.4598 | True              | False     |                      8.5007 |
