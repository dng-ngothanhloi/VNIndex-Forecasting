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
| ARDL_AIC            | 5392.222363387498  | 5390.09705358187   |
| ARDL_BIC            | 5414.637900144784  | 5421.47880504207   |
| ARDL_HQIC           | 5400.914362637223  | 5402.265852531485  |
| ARDL_RMSE_trainval  | 30.37650043922527  | 30.233063041475162 |
| ARDL_RMSE_test      | 14.064042923175688 | 13.97184196091624  |
| ARDL_R2_trainval    | 0.9497513532629133 | 0.9502247796343455 |
| ARDL_R2_test        | 0.847587230115677  | 0.8495790529698604 |
| ARDL_durbin_watson  | 1.9479673077766593 | 1.934520101426282  |
| ARDL_ljungbox_q10   | 5.539177961969757  | 6.258971405496287  |
| ARDL_ljungbox_p     | 0.8523811261190686 | 0.7930565003231778 |
| LSTM_lookback       | 10                 | 60                 |
| LSTM_batch_size     | 8                  | 16                 |
| LSTM_best_epoch     | 145                | 120                |
| LSTM_input_channels | 3                  | 5                  |
| LSTM_units          | [32, 16]           | [32, 16]           |
| LSTM_dense_units    | [8]                | [8]                |
| LSTM_dropout        | 0.1                | 0.1                |
| LSTM_lr             | 0.0003             | 0.0003             |
| LSTM_max_epochs     | 300                | 300                |
| LSTM_es_patience    | 30                 | 30                 |
| LSTM_rlr_patience   | 12                 | 12                 |
| LSTM_Val_RMSE       | 17.894608574525332 | 25.211405717815865 |
| LSTM_Train_RMSE     | 19.573100512446462 | 24.82001490782936  |
| LSTM_Test_RMSE      | 21.695154553316176 | 27.40372397947574  |
| LSTM_train_samples  | 526                | 476                |
| LSTM_val_samples    | 123                | 123                |
