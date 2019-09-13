#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from subprocess import call
import csv
from collections import Counter
import numpy as np
from sklearn import preprocessing

#%%
# features: bot left corner, bot right corner, bot center, more in center, more pieces on board
# todo are we looking at strings or integers when we read in the board

# construct df
def constructDF():
    # read in csv
    with open('trainDataSet (2).csv', newline='') as csvfile:
        boards = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(boards) # skip first line

        data = []
        for board in boards:
            blc = botLeftCorner(board)
            brc = botRightCorner(board)
            bc = botCenter(board)
            mc = moreInCenter(board)
            mb = moreOnBoard(board)
            winner = board[42]

            data.append([blc, brc, bc, mc, mb, winner])
    return pd.DataFrame(data, columns=['Bot Left Corner', 'Bot Right Corner', 'Bot Center', 'More in Center', 'More on Board', 'Winner'])

# return 0 if nobody is occupying the bottom left corner, otherwise 1 or 2 for the player that is occupying it
def botLeftCorner(board):
    return board[0]

# return 0 if nobody is occupying the bottom right corner, otherwise 1 or 2 for the player that is occupying it
def botRightCorner(board):
    return board[36]

# return 0 if nobody is occupying the bottom center, otherwise 1 or 2 for the player that is occupying it
def botCenter(board):
    return board[18]

# return 'tie' if they both have the same number of pieces in the center, otherwise '1' or '2' for the player with more
def moreInCenter(board):
    num1 = 0
    num2 = 0
    # center indexes: 18-23
    for i in range (18, 24):
        piece = board[i]
        if piece == '1':
            num1 += 1
        elif piece == '2':
            num2 += 1

    if num1 > num2:
        return '1'
    elif num2 > num1:
        return '2'
    else:
        return 'tie'

# return 'tie' if they both have the same number of pieces on the board, otherwise '1' or '2' for the player with more
def moreOnBoard(board):
    counts = Counter(board)
    num1 = counts['1']
    num2 = counts['2']
    
    if num1 > num2:
        return '1'
    elif num2 > num1:
        return '2'
    else:
        return 'tie'

#%%
df = constructDF()
# one hot encode our data
df = pd.get_dummies(df)

# x and y dataset
# here we want to predict if 1 or 2 wins, major todo here
x = df.drop(['Winner_1', 'Winner_2'], axis=1)
print(x.head())
# x = x.drop('More in Center', axis=1)
# x = x.drop('More on Board', axis=1)
y = df['Winner_1']

# test to see what our dataframes look like
df.to_csv('df.csv')
x.to_csv('x.csv')
y.to_csv('y.csv')

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

#%%
# create our decision tree classifier
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print('Accuracy score:', accuracy_score(y_test, y_predict))

# matrix = pd.DataFrame(
#     confusion_matrix(y_test, y_predict),
#     columns=['Predicted Loss', 'Predicted Win'],
#     index=['True Loss', 'True Win']
# )
# print(matrix)

# export to a tree
tree.export_graphviz(model, out_file='tree.dot')
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

#%%
# here is some test code, seperate from everything above
# i am using it to test one hot encoding our data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from subprocess import call

data = pd.DataFrame()
data['A'] = ['a','a','b','a', 'a', 'a']
data['B'] = ['b','b','a','b', 'b', 'a']
data['C'] = ['2', '0', '1', '0', '2', '2']
data['Class'] = ['n','n','y','n', 'y', 'y']

t = DecisionTreeClassifier()

one_hot_data = pd.get_dummies(data[['A','B','C']])
print(one_hot_data.head())
t.fit(one_hot_data, data['Class'])
tree.export_graphviz(t, out_file='tree2.dot')
call(['dot', '-T', 'png', 'tree2.dot', '-o', 'tree2.png'])

#%%
