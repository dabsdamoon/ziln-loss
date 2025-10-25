# Data Explanation

## Fields

history
id - A unique id representing a customer
chain - An integer representing a store chain
offer - An id representing a certain offer
market - An id representing a geographical region
repeattrips - The number of times the customer made a repeat purchase
repeater - A boolean, equal to repeattrips > 0
offerdate - The date a customer received the offer

transactions
id - see above
chain - see above
dept - An aggregate grouping of the Category (e.g. water)
category - The product category (e.g. sparkling water)
company - An id of the company that sells the item
brand - An id of the brand to which the item belongs
date - The date of purchase
productsize - The amount of the product purchase (e.g. 16 oz of water)
productmeasure - The units of the product purchase (e.g. ounces)
purchasequantity - The number of units purchased
purchaseamount - The dollar amount of the purchase

offers
offer - see above
category - see above
quantity - The number of units one must purchase to get the discount
company - see above
offervalue - The dollar value of the offer
brand - see above

## Preprocessing

### Purpose
- Predicting each customer’s total purchase value in the next 12 months following the initial purchase
- Model features include the initial purchase amount, the number of items purchased, as well as the store chain, product category, product brand, and product size measure of each individual purchased item.

### Details
- Use only 20 parquets with chunk size of 1,000,000
- For each company, randomly pick 80% of customers for model training and use the remaining 20% for model evaluation.