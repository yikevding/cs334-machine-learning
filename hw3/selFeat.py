import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler



def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """

    weekday=[]
    peak=[]
    for index, row in df.iterrows():
        timestamp=row["date"].split(" ")
        date=timestamp[0]
        hour=timestamp[1].split(":")[0]
        year=date[-2:]
        monthday=date[0:len(date)-len(year)]
        year='20'+year
        date_str=monthday+year
        format_str=datetime.strptime(date_str,"%m/%d/%Y")
        if(format_str.isoweekday()==6 or format_str.isoweekday()==7):
            weekday.append(0)
        else:
            weekday.append(1)
        if(int(hour)>=14 and int(hour)<=23): # suppose peak hour is between 14:00-23:00
            peak.append(1)
        else:
            peak.append(0)
    df["weekday"]=weekday
    df["peak"]=peak
    df = df.drop(columns=['date'])

    return df


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """

    # basically i take out attribute that have >0.6 with other attributes
    selected=pd.DataFrame()
    selected["lights"]=df["lights"]
    selected["Press_mm_hg"]=df["Press_mm_hg"]
    selected["RH_out"]=df["RH_out"]
    selected["Windspeed"]=df["Windspeed"]
    selected["Visibility"]=df["Visibility"]
    selected["weekday"]=df["weekday"]
    selected["peak"]=df["peak"]
    selected["T_out"]=df["T_out"]
    selected["T1"]=df["T1"]
    selected["RH_2"]=df["RH_2"]
    selected["RH_5"]=df["RH_5"]
    selected["RH_6"]=df["RH_6"]
    df=selected
    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """

    # apply standardization to all x variables
    # not apply transformation to categorical variables
    stdScaler=StandardScaler()
    weekdayTrain=trainDF["weekday"]
    weekdayTest=testDF["weekday"]
    peakTrain=trainDF["peak"]
    peakTest=testDF["peak"]

    trainDF=trainDF.drop(columns=["weekday"])
    trainDF=trainDF.drop(columns=["peak"])
    testDF=testDF.drop(columns=["weekday"])
    testDF=testDF.drop(columns=["peak"])

    stdScaler.fit(trainDF[trainDF.columns])
    trainDF[trainDF.columns]=stdScaler.transform(trainDF[trainDF.columns])
    testDF[testDF.columns]=stdScaler.transform(testDF[testDF.columns])

    trainDF["weekday"]=weekdayTrain
    trainDF["peak"]=peakTrain
    testDF["weekday"]=weekdayTest
    testDF["peak"]=peakTest

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
