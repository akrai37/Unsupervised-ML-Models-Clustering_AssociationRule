import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset=pd.read_csv('Market_Basket_Optimisation.csv', header=None)
#transactions=dataSet.values.tolist()
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)]) 

#training apriori fro dataset
from apyori import apriori    
rules=apriori(transactions ,min_support=0.003, min_confidence=0.2, min_lift=3)

#visualizing the result
results=list(rules) 

'''myResults=[list(x) for x in results]'''  
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
    
'''def inspect(results):
    rh          = [tuple(result[2][0][0])[0] for result in results]
    lh          = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

pd.DataFrame(inspect(results))'''

'''def display_top_products(results, n_products=5):
    print("Support\tConf.\tLift\tProducts")
    for result in results[:n_products]:
        support = round(100 * result.support, 2)
        confidence = round(result.ordered_statistics[0].confidence, 2)
        lift = round(result.ordered_statistics[0].lift, 2)
        products = " + ".join(list(result.items))
        print("{0}%\t{1}\t{2}\t{3}".format(support, confidence, lift, products))
display_top_products(results, 20)'''





    