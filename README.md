# Wavelet Transform based Seq2Seq Model for Time Series Forecasting

This is a instance for sequence to sequence model for time series forecasting, including the straightaway implement of MLP, RNN, CNN, LSTM, GRU, TCN, VAR, Wavelet (which converts the 1-dim tiem-series into 2-dim time-frequency diagram then put the Seq2Seq model on it for forecasting) with Pytorch.

## Platform

* python 3.x
* Pytorch 0.4+

## Running

take the function LetsGo()  in 'main.py' with parameter 'train' for training the model and 'pre' for testing the model with the trained models' paths with parameter m_path.

## dataset

[ENSO](https://www.esrl.noaa.gov/psd/data/climateindices/): NINO1-2, NINO3, NINO3-4, NINO 4 


## License

This project is licensed under the MIT License - see the [LICENSE.md] file for details


