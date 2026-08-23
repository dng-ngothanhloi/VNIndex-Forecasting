**Table 6. Exact trained configuration at CEV=0.75 and CEV=0.85**

| Parameter           | CEV_0.75           | CEV_0.85           |
|:--------------------|:-------------------|:-------------------|
| label               | pca_cev_0.75       | pca_cev_0.85       |
| k                   | 2.0                | 4.0                |
| ARDL_order_P        | 1                  | 1                  |
| ARDL_order_Q        | 1                  | 1                  |
| ARDL_causal         | True               | True               |
| ARDL_hold_back      | 5                  | 5                  |
| ARDL_num_params     | 4                  | 6                  |
| ARDL_nobs           | 654                | 654                |
| ARDL_selection      | BIC                | BIC                |
| ARDL_AIC            | 5392.222363387498  | 5390.097053581868  |
| ARDL_BIC            | 5414.637900144784  | 5421.478805042068  |
| ARDL_HQIC           | 5400.914362637223  | 5402.265852531483  |
| ARDL_RMSE_trainval  | 30.376500439225094 | 30.23306304147518  |
| ARDL_RMSE_test      | 14.064042923175661 | 13.971841960916175 |
| ARDL_R2_trainval    | 0.9497513532629138 | 0.9502247796343455 |
| ARDL_R2_test        | 0.8475872301156776 | 0.8495790529698617 |
| ARDL_durbin_watson  | 1.9479673077766528 | 1.9345201014262758 |
| ARDL_ljungbox_q10   | 5.539177961970093  | 6.2589714054968235 |
| ARDL_ljungbox_p     | 0.8523811261190427 | 0.7930565003231309 |
| LSTM_lookback       | 60                 | 20                 |
| LSTM_batch_size     | 32                 | 16                 |
| LSTM_best_epoch     | 147                | 50                 |
| LSTM_input_channels | 3                  | 5                  |
| LSTM_units          | [64, 32]           | [64, 32]           |
| LSTM_dense_units    | [16]               | [16]               |
| LSTM_dropout        | 0.2                | 0.2                |
| LSTM_lr             | 0.0001             | 0.0001             |
| LSTM_max_epochs     | 150                | 150                |
| LSTM_es_patience    | 25                 | 25                 |
| LSTM_rlr_patience   | 10                 | 10                 |
| LSTM_Val_RMSE       | 25.72765362870104  | 25.127948543347458 |
| LSTM_Train_RMSE     | 31.27255353629998  | 32.31708779914208  |
| LSTM_Test_RMSE      | 25.44101780569286  | 33.42357484708038  |
| LSTM_train_samples  | 476                | 516                |
| LSTM_val_samples    | 123                | 123                |
