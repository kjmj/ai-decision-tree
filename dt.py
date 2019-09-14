import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from subprocess import call
import csv
import sys

# extract features and return a data frame with those features
def extract_features(input_file):
    # read in csv
    with open(input_file, newline='') as csvfile:
        boards = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(boards) # skip first line

        data = []
        for board in boards:
            blc = bot_left_corner(board)
            brc = bot_right_corner(board)
            bc = bot_center(board)
            mc = more_in_center(board)
            winner = board[42]

            data.append([blc, brc, bc, mc, winner])
    return pd.DataFrame(data, columns=['Bot Left Corner', 'Bot Right Corner', 'Bot Center', 'More in Center', 'Winner'])

# return 0 if nobody is occupying the bottom left corner, otherwise 1 or 2 for the player that is occupying it
def bot_left_corner(board):
    return board[0]

# return 0 if nobody is occupying the bottom right corner, otherwise 1 or 2 for the player that is occupying it
def bot_right_corner(board):
    return board[36]

# return 0 if nobody is occupying the bottom center, otherwise 1 or 2 for the player that is occupying it
def bot_center(board):
    return board[18]

# return 'tie' if they both have the same number of pieces in the 3 center columns, otherwise '1' or '2' for the player with more
def more_in_center(board):
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


# assignment specifies data input is first arg, feature export is second arg
input_file = sys.argv[1]
output_file = sys.argv[2]

# extract the features and construct a dataframe
df = extract_features(input_file)

# one hot encode our data
df = pd.get_dummies(df)

# export our features csv
df.to_csv(output_file)

# seperate into data and known truths
x = df.drop(['Winner_1', 'Winner_2'], axis=1)
y = df['Winner_1']

# test to see what our dataframes look like
# df.to_csv('df.csv')
# x.to_csv('x.csv')
# y.to_csv('y.csv')

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# create our decision tree classifier
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print('Accuracy score:', accuracy_score(y_test, y_predict))

matrix = pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Loss', 'Predicted Win'],
    index=['True Loss', 'True Win']
)
print(matrix)

# export the tree to an image
tree.export_graphviz(model, out_file='tree.dot', feature_names=x.columns)
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

# cross validation
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('3 fold cross validation score:', results)