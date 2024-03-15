#!/usr/bin/env python
# coding: utf-8

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd



def plot_network(model):
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3500, node_color='skyblue', font_size=10, font_weight='bold', arrowstyle='-|>', arrowsize=12)
        plt.title('Bayesian Network Visualization', size=15)
        plt.show()

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork(
    [
        ('DebtIncomeRatio', 'PaymentHistory'),
        ('PaymentHistory', 'Age'),
        ('PaymentHistory', 'Reliability'),
        ('Age', 'Reliability'),
        ('Income','Assets'), 
        ('Income', 'FutureIncome'),
        ('Assets', 'FutureIncome'),
        ('DebtIncomeRatio', 'BankLoan'),
        ('Reliability', 'BankLoan'),
        ('FutureIncome', 'BankLoan')
    ]
)

# Defining individual CPDs.
cpd_DebtIncomeRatio = TabularCPD(variable='DebtIncomeRatio', variable_card=2, values=[[0.5], [0.5]], state_names={'DebtIncomeRatio': ['Low', 'High']})
cpd_Income = TabularCPD(variable='Income', variable_card=3, values=[[0.333], [0.333], [0.334]], state_names={'Income': ['High', 'Medium', 'Low']})


# The representation of CPD in pgmpy is a bit different than the CPD shown in the above picture. In pgmpy the colums
# are the evidences and rows are the states of the variable. So the grade CPD is represented like this:
#
#    +----------------+----------------+----------------+-----------+----------------+----------------+------------+----------------+----------------+--------------+
#    | PaymentHistory |   Excellent    |    Excellent   | Excellent |   Acceptable   |   Acceptable   | Acceptable |  Unacceptable  |  Unacceptable  | Unacceptable |
#    +----------------+----------------+----------------+-----------+----------------+----------------+------------+----------------+----------------+--------------+
#    | Age            | Between16and25 | Between26and64 |   Over65  | Between16and25 | Between26and64 |   Over65   | Between16and25 | Between26and64 |    Over65    |
#    +----------------+----------------+----------------+-----------+----------------+----------------+------------+----------------+----------------+--------------+
#    | Reliability_0  |       0.7      |       0.8      |    0.9    |       0.6      |       0.7      |     0.8    |       0.5      |       0.6      |      0.7     |
#    +----------------+----------------+----------------+-----------+----------------+----------------+------------+----------------+----------------+--------------+
#    | Reliability_1  |       0.3      |       0.2      |    0.1    |       0.4      |       0.3      |     0.2    |       0.5      |       0.4      |      0.3     |
#    +----------------+----------------+----------------+-----------+----------------+----------------+------------+----------------+----------------+--------------+


cpd_Reliability = TabularCPD(variable='Reliability', variable_card=2,
                      values=[[0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7],
                              [0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3],],
                      evidence=['PaymentHistory','Age'],
                      evidence_card=[3,3],
                      state_names={'Reliability': ['Reliable', 'Unreliable'],
                                   'PaymentHistory': ['Excellent', 'Acceptable', 'Unacceptable'],
                                   'Age': ['Between16and25', 'Between26and64', 'Over65']})

cpd_Age = TabularCPD(variable='Age', variable_card=3, 
                      values=[[0.1, 0.333, 0.6],
                              [0.3, 0.333, 0.3],
                              [0.6, 0.334, 0.1]],
                      evidence=['PaymentHistory'],
                      evidence_card=[3],
                      state_names={'Age': ['Between16and25', 'Between26and64', 'Over65'],
                                   'PaymentHistory': ['Excellent', 'Acceptable', 'Unacceptable']})


cpd_PaymentHistory = TabularCPD(variable='PaymentHistory', variable_card=3, 
                      values=[[0.6, 0.1],
                              [0.3, 0.3],
                              [0.1, 0.6]],
                      evidence=['DebtIncomeRatio'],
                      evidence_card=[2],
                      state_names={'PaymentHistory': ['Excellent', 'Acceptable', 'Unacceptable'],
                                   'DebtIncomeRatio': ['Low', 'High']})

cpd_BankLoan = TabularCPD(variable='BankLoan', variable_card=2, 
                      values=[[0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2],
                              [0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8]],
                      evidence=['DebtIncomeRatio','Reliability','FutureIncome'],
                      evidence_card=[2,2,2],
                      state_names={'BankLoan': ['Positive', 'Negative'],
                                   'DebtIncomeRatio': ['Low', 'High'],
                                   'Reliability': ['Reliable', 'Unreliable'],
                                   'FutureIncome': ['Promising', 'Not_promising']})

# {'DebtIncomeRatio': ['Low', 'High']})
# {'Income': ['High', 'Medium', 'Low']})
# {'PaymentHistory': ['Excellent', 'Acceptable', 'Unacceptable']}
# {'Age': ['Between16and25', 'Between26and64', 'Over65']}
# {'Reliability': ['Reliable', 'Unreliable']}
# {'Assets': ['High', 'Medium', 'Low']}
# {'FutureIncome': ['Promising', 'Not_promising']}
# {'BankLoan': ['Positive', 'Negative']}

cpd_Assets = TabularCPD(variable='Assets', variable_card=3,
                        values=[[0.7, 0.2, 0.1], 
                                [0.2, 0.7, 0.2], 
                                [0.1, 0.1, 0.7]], 
                        evidence=['Income'], 
                        evidence_card=[3],
                        state_names={'Assets': ['High', 'Medium', 'Low'],
                                     'Income': ['High', 'Medium', 'Low']})


cpd_FutureIncome = TabularCPD(variable='FutureIncome', variable_card=2,
                              values=[[0.9, 0.8, 0.7, 0.7, 0.6, 0.4, 0.5, 0.4, 0.2],
                                      [0.1, 0.2, 0.3, 0.3, 0.4, 0.6, 0.5, 0.6, 0.8]],
                              evidence=['Income', 'Assets'],
                              evidence_card=[3, 3], 
                              state_names={'FutureIncome': ['Promising', 'Not_promising'],# les lignes
                                           'Income': ['High', 'Medium', 'Low'],
                                           'Assets': ['High', 'Medium', 'Low']})


model.add_cpds(cpd_DebtIncomeRatio, cpd_Income, cpd_PaymentHistory, cpd_Age, cpd_Reliability, cpd_Assets, cpd_FutureIncome, cpd_BankLoan)
print(model.check_model())

# draw  the network
# plot_network(model)

# Itérer chaque noeud du model pour trouver  ses dépendances locales
for node in model.nodes():
    print(f"Indépendances locales de {node} :",model.local_independencies(node))
    print()


# Chaînes actives à partir de "Income"
active_trails_income = model.active_trail_nodes('Income')
print("Chaînes actives à partir de 'Income' : ",active_trails_income)

# Chaînes actives à partir de "Income" en observant "BankLoan"
active_trails_income_given_bankloan = model.active_trail_nodes('Income', observed='BankLoan')
print ("Chaînes actives à partir de ""Income"" en observant ""BankLoan"" : " , active_trails_income_given_bankloan)


# Charger les données
data = pd.read_csv('50000-cases.csv')
# Apprendre les CPDs à partir des données
model.fit(data, estimator=MaximumLikelihoodEstimator)

cpd_future_income = model.get_cpds('FutureIncome')
cpd_assets = model.get_cpds('Assets')
print("CPD de FutureIncome :")
print(cpd_future_income)
print("\nCPD de Assets :")
print(cpd_assets)

from pgmpy.inference import VariableElimination

# Créer l'objet pour l'inférence
inference = VariableElimination(model)

# P(BankLoan)
prob_bank_loan = inference.query(variables=['BankLoan'])
print("P(BankLoan):")
print(prob_bank_loan)

# P(BankLoan|Income = Low, Age = Between16and25, PaymentHistory = Excellent, Assets = Low)
prob_bank_loan_given_conditions1 = inference.query(
    variables=['BankLoan'],
    evidence={'Income': 'Low', 'Age': 'Between16and25', 'PaymentHistory': 'Excellent', 'Assets': 'Low'}
)
print("\nP(BankLoan|Income = Low, Age = Between16and25, PaymentHistory = Excellent, Assets = Low):")
print(prob_bank_loan_given_conditions1)

# P(BankLoan|Income = High, Age = Between16and25, PaymentHistory = Excellent, Assets = High)
prob_bank_loan_given_conditions2 = inference.query(
    variables=['BankLoan'],
    evidence={'Income': 'High', 'Age': 'Between16and25', 'PaymentHistory': 'Excellent', 'Assets': 'High'}
)
print("\nP(BankLoan|Income = High, Age = Between16and25, PaymentHistory = Excellent, Assets = High):")
print(prob_bank_loan_given_conditions2)

# P(BankLoan|Income = High, Age = Over65, PaymentHistory = Excellent, Assets = High)
prob_bank_loan_given_conditions3 = inference.query(
    variables=['BankLoan'],
    evidence={'Income': 'High', 'Age': 'Over65', 'PaymentHistory': 'Excellent', 'Assets': 'High'}
)
print("\nP(BankLoan|Income = High, Age = Over65, PaymentHistory = Excellent, Assets = High):")
print(prob_bank_loan_given_conditions3)

from pgmpy.sampling import BayesianModelSampling

# Créer un objet de sampling
sampler = BayesianModelSampling(model)

# Générer des échantillons
samples = sampler.forward_sample(size=10000)

# P(BankLoan)
prob_bank_loan_sampled = samples['BankLoan'].value_counts() / samples.shape[0]
print("P(BankLoan) estimée par échantillonnage :")
print(prob_bank_loan_sampled)


# Filtrer les échantillons pour correspondre à l'évidence
filtered_samples = samples[
    (samples['Income'] == 'Low') &
    (samples['Age'] == 'Between16and25') &
    (samples['PaymentHistory'] == 'Excellent') &
    (samples['Assets'] == 'Low')
]

# Calculer la distribution conditionnelle de BankLoan
prob_bank_loan_given_conditions_sampled = filtered_samples['BankLoan'].value_counts() / filtered_samples.shape[0]
print("\nP(BankLoan|Income = Low, Age = Between16and25, PaymentHistory = Excellent, Assets = Low) estimée par échantillonnage :")
print(prob_bank_loan_given_conditions_sampled)
