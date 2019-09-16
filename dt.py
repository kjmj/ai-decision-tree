import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from subprocess import call
import csv
import sys
import pydot

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
            ml = more_in_leftmost(board)
            mr = more_in_rightmost(board)
            winner = board[42]

            data.append([blc, brc, bc, ml, mr, mc, winner])
    return pd.DataFrame(data, columns=['Bot Left Corner', 'Bot Right Corner', 'Bot Center', 'More in Leftmost', 'More in Rightmost', 'More in Center', 'Winner'])

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

# return 'tie' if they both have the same number of pieces in the leftmost column, otherwise '1' or '2' for the player with more
def more_in_leftmost(board):
    num1 = 0
    num2 = 0
    # leftmost indexes: 0-5
    for i in range (0,5):
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

# return 'tie' if they both have the same number of pieces in the rightmost column, otherwise '1' or '2' for the player with more
def more_in_rightmost(board):
    num1 = 0
    num2 = 0
    # leftmost indexes: 36-41
    for i in range (36,41):
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
y = df['Winner_2']

# split our data into 60% test, 20% train, and 20% validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

# create our decision tree classifier
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('Accuracy score single tree:', accuracy_score(y_test, y_predict))

# export the tree to an image
dot = tree.export_graphviz(model, feature_names=x.columns)
(graph,)=pydot.graph_from_dot_data(dot)
graph.write_png("tree.png")

# get feature importances
feature_importances = pd.DataFrame.from_dict([dict(zip(x.columns, model.feature_importances_))])
feature_importances = feature_importances.transpose().reset_index()
feature_importances.columns = ['feature', 'feature importance']
feature_importances.to_csv('feature_importances_tree.csv', index=False)

# cross validation
kfold = KFold(n_splits=3, shuffle=True)
train_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
val_results = cross_val_score(model, x_val, y_val, cv=kfold, scoring='accuracy')
print('3-fold training results:', train_results)
print('3-fold validation results:', val_results)

# random forest experiment
forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
forest = forest.fit(x_train, y_train)
y_predict = forest.predict(x_test)
print('Accuracy score random forest:', accuracy_score(y_test, y_predict))
feature_importances_forest = pd.DataFrame.from_dict([dict(zip(x.columns, forest.feature_importances_))])
feature_importances_forest = feature_importances_forest.transpose().reset_index()
feature_importances_forest.columns = ['feature', 'feature importance']
feature_importances_forest.to_csv('feature_importances_forest.csv', index=False)
