# specialty-classification-medical-transcriptions

LSTM -

This code preprocesses medical transcriptions by cleaning text, removing stopwords, and applying lemmatization. It extracts keywords using the YAKE algorithm and encodes medical specialties as labels. The dataset is split into training and testing sets, with tokenized and padded sequences for deep learning. A bidirectional LSTM model with dropout, batch normalization, and regularization is built to classify medical specialties. The model is trained using sparse categorical cross-entropy loss and the Adam optimizer. Finally, the model is evaluated on the test set to measure accuracy. This approach improves text classification by focusing on key phrases rather than full transcriptions.
