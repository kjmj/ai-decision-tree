
"""
This is a test program to show you guys how to make sure 
that you can pass in arguments on the command line so that 
any OS can run your code properly.
Here is a good link:
https://stackoverflow.com/questions/1009860/how-to-read-process-command-line-arguments

Python 2.7 link:

https://stackoverflow.com/questions/14471049/python-2-7-1-how-to-open-edit-and-close-a-csv-file

Python 3 link:

https://stackoverflow.com/questions/34283178/typeerror-a-bytes-like-object-is-required-not-str-in-python-and-csv

"""

import sys # This is built into python
import csv # Also built in to pyhthon

"""
This program wants a number, an input *.csv file, and an output *.csv file.

To run this in Windows, you will type this into your command prompt:

python argumenttester.py 1 takemeon.csv rejectme.csv

"""

if __name__ == "__main__":
    # These 3 if statements read in the first argument denoted by "argv[1]"
    if (sys.argv[1] == '1'):
        print("You passed in a 1!")
    elif (sys.argv[1] == '2'):
        print("You passed in a 2!")
    else:
        print("You passed in: ")
        print(sys.argv[1])
    
    """
    We now want to read in a second file that is a *.csv file.

    I am not sure what the course staff think, but the current
    instructions suggest that you are required to use python 
    for this project. They have not decided how to fix this 
    issue yet, so here is a quick python tutorial. Maybe I 
    can create a java or cpp tutorial later.
    """

    # The *.csv files
    first_input_csv = sys.argv[2]
    """ This should be in the same directory as this file,
    otherwise you have to give the absolute path.
    """
    second_output_csv = sys.argv[3] # Same here

    # Now we want to open the *.csv file.
    inputcsv = open(first_input_csv, 'r') # This is means we are openning a file and then editing it.
    # 'rb' is for python2. 'r' is python 3
    inputreader = csv.reader(inputcsv) # Now we can read the file that was opened by python

    print("You read me like a book.")

    """
    Now you can do whatever you want to the input file.
    """

    """
    Now we want to create a file with the second filename.
    This is how I am doing it. 
    """

    # Here are some changes we are going to make/create
    print("We will write these changes to " +str(second_output_csv))
    changes = [    
    ['1 dozen','12'],                                                            
    ['1 banana','13'],                                                           
    ['1 dollar','elephant','heffalump'],
    ]
    print(changes)
    
    """
    Now we will write the changes to the output file.
    
    First, we need to take in the string that is the 
    filename and create a filepath out of it.

    """

    # Now, if the rejectme.csv file is being viewed in the GUI
    # before the script runs, you will get write permission errors.

    # Creating the path
    outfilepath = "./" + str(second_output_csv)
    # Opening the file at the path.
    outfile = open(outfilepath,'w')
    # Then we create an object to write data with.
    writer = csv.writer(outfile, delimiter=',')
    # Now we do the writing
    for row in changes:
        writer.writerow(row)
    print("You win!!")
    """
    Now you have read in and written to a *.csv file
    the way the graders will.
    
    """
