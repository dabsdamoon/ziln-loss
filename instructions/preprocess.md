5.1 KAGGLE ACQUIRE VALUED SHOPPERS CHALLENGE
The dataset for the Kaggle Acquire Valued Shoppers Challenge competition contains complete
basket-level shopping history for 311K customers from 33K companies. We consider the task of
predicting each customer’s total purchase value in the next 12 months following the initial purchase.
Model features include the initial purchase amount, the number of items purchased, as well as the
8
store chain, product category, product brand, and product size measure of each individual purchased
item.
We restrict our experiment to the top twenty companies based on customer count and focus on the
cohort of customers who first purchased between 2012-03-01 and 2012-07-01. For each company, we
randomly pick 80% of customers for model training and use the remaining 20% for model evaluation.
We conduct our experiment along two axes: model architecture and loss. Both linear and DNN
model are considered. The ZILN loss is compared to the MSE loss. We additionally report the binary
classification results of returning customer prediction.
We implement our models using the TensorFlow framework. Following standard practice, for
categorical features, we use one-hot encodings in linear models and embeddings in DNN. For DNN,
we consider two hidden layers with 64 and 32 number of units, respectively. We train each model for
up to 400 epochs with a batch size of 1,024 and the Adam optimizer (Kingma & Ba, 2014) with a
learning rate of 2e-4. We also apply an early stopping rule to prevent overfitting.
