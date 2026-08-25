##  Scenaro 1: lookback_values: [5,10,15,20,30]
### Runing CEV 0.75 Same with original (python experiments/run_experiment.py --reduction pca --cev 0.75 --include-multiseed 
Phase 3D] Per-seed Test metrics:
 Seed  N_Test      RMSE       MAE  MAPE(%)       R2
   42     167 26.482501 14.872046 1.206868 0.459595
   52     167 26.122360 14.173379 1.153060 0.474194
   62     167 26.067003 14.715432 1.192623 0.476420
   72     167 27.723697 15.631363 1.272769 0.407752
   82     167 26.487191 15.235689 1.234472 0.459404

[Phase 3D] Mean +/- Std summary (n=5 seeds):
  RMSE    : 26.5766 +/- 0.6706  (min=26.0670, max=27.7237)
  MAE     : 14.9256 +/- 0.5493  (min=14.1734, max=15.6314)
  MAPE(%) : 1.2120 +/- 0.0449  (min=1.1531, max=1.2728)
  R2      : 0.4555 +/- 0.0278  (min=0.4078, max=0.4764)

### Runing CEV 0.85 Same with original (python experiments/run_experiment.py --reduction pca --cev 0.85 --include-multiseed )

Summary Results:
   Lookback  Batch_size  Train_RMSE   Train_MAE  Train_MAPE(%)   Val_RMSE    Val_MAE  Val_MAPE(%)  Train_samples  Val_samples Train_period_start  \
0         5          16   27.537592   20.779023       1.798795  21.089037  16.627632     1.333381            531          123         2022-01-11   
1         5          32  138.253332  109.615866       9.176318  76.647388  72.011713     5.683388            531          123         2022-01-11   
2        10          16  130.472809  104.411877       8.809140  77.465077  72.904487     5.754474            526          123         2022-01-18   
3        10          32  133.490667  106.542256       8.983722  76.765377  72.164203     5.695633            526          123         2022-01-18   
4        15          16  127.879335  102.391867       8.677917  77.735650  73.201429     5.778128            521          123         2022-01-25   
5        15          32  131.316721  104.872222       8.884546  76.715794  72.131592     5.693132            521          123         2022-01-25   
6        20          16   32.317088   24.286476       2.110041  25.127949  20.063607     1.611230            516          123         2022-02-08   
7        20          32   37.869374   28.944937       2.498193  25.796914  20.417945     1.641259            516          123         2022-02-08   
8        30          16  121.160805   97.423277       8.397535  77.280578  72.743014     5.741792            506          123         2022-02-22   
9        30          32  124.630039   99.956323       8.614016  76.294462  71.711625     5.659868            506          123         2022-02-22   

  Train_period_end Val_period_start Val_period_end  Best_Epoch  
0       2024-02-29       2024-03-01     2024-08-26          38  
1       2024-02-29       2024-03-01     2024-08-26           1  
2       2024-02-29       2024-03-01     2024-08-26           1  
3       2024-02-29       2024-03-01     2024-08-26           1  
4       2024-02-29       2024-03-01     2024-08-26           1  
5       2024-02-29       2024-03-01     2024-08-26           1  
6       2024-02-29       2024-03-01     2024-08-26          50  
7       2024-02-29       2024-03-01     2024-08-26          60  
8       2024-02-29       2024-03-01     2024-08-26           1  
9       2024-02-29       2024-03-01     2024-08-26           1  

[SELECT] Selected model by Val_RMSE: lookback=5, batch=16  Val_RMSE=21.0890  best_epoch=38
[SAVED] selected_tuning_history.csv (LB5_BS16)

[FINAL REFIT] lookback=5 batch=16 epochs=38 on Train+Val — NO EarlyStopping / ReduceLROnPlateau
[FINAL] Test — RMSE: 29.35463  MAE: 17.28824  MAPE: 1.40610%
[FINAL] Train+Val (dev) fit — RMSE: 26.16906  MAE: 19.79253  MAPE: 1.69241%



==============================================================================
 MULTI-SEED STABILITY: lookback=5 batch=16 epochs=38 (FROZEN from seed=42 reference tuning) 
==============================================================================

[SEED 42] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...
[SEED 42] Test RMSE=29.3546 MAE=17.2882 MAPE=1.4061% R2=0.3360

[SEED 52] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...
[SEED 52] Test RMSE=28.1724 MAE=17.2356 MAPE=1.3936% R2=0.3884

[SEED 62] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...
[SEED 62] Test RMSE=29.8823 MAE=17.1964 MAPE=1.4011% R2=0.3119

[SEED 72] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...
[SEED 72] Test RMSE=25.8778 MAE=15.3579 MAPE=1.2387% R2=0.4840

[SEED 82] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...
[SEED 82] Test RMSE=27.5061 MAE=15.4664 MAPE=1.2578% R2=0.4170

[Phase 3D] Per-seed Test metrics:
 Seed  N_Test      RMSE       MAE  MAPE(%)       R2
   42     167 29.354631 17.288239 1.406102 0.336021
   52     167 28.172374 17.235585 1.393644 0.388428
   62     167 29.882295 17.196414 1.401142 0.311936
   72     167 25.877839 15.357927 1.238675 0.483991
   82     167 27.506144 15.466433 1.257849 0.417011

[Phase 3D] Mean +/- Std summary (n=5 seeds):
  RMSE    : 28.1587 +/- 1.5835  (min=25.8778, max=29.8823)
  MAE     : 16.5089 +/- 1.0024  (min=15.3579, max=17.2882)
  MAPE(%) : 1.3395 +/- 0.0837  (min=1.2387, max=1.4061)
  R2      : 0.3875 +/- 0.0681  (min=0.3119, max=0.4840)

