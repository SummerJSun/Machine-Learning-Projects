import argparse
import pandas as pd


def extract_date(df):
    """
    Extract relevant features from the DATE column and drop the column
    
    Parameters
    ----------
    df : pandas.DataFrame
        the original pandas dataframe

    Returns
    -------
    df : pandas.DataFrame
        return the pandas dataframe with DATE column dropped and 
        new features added to it.
    """
    return df


def extract_company(df):
    """
    Extract relevant features from the STOCK column and drop the column
    
    Parameters
    ----------
    df : pandas.DataFrame
        the original pandas dataframe

    Returns
    -------
    df : pandas.DataFrame
        return the pandas dataframe with STOCK column dropped and 
        new features added to it.
    """
    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the features
    
    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe

    Returns
    -------
    trainDF : pandas.DataFrame
        return the transformed trainDF dataframe
    testDF : pandas.DataFrame
        return the transformed testDT dataframe
    """
    return trainDF, testDF


def extract_binary(trainseries, testseries, k):
    """
    Extract relevant binary word features from the tweet features.
    It is important to note the top k words are taken from the trainseries
    so may not be the top k words in the testseries!
    
    Parameters
    ----------
    trainseries : pandas.Series
        the input tweet features from the training data (i.e., TWTOKEN)
    testseries : pandas.Series
        the input tweet features from the test data (i.e., TWTOKEN)   
    k : int
        the number of top k words (in terms of term frequency) from the 
        training corpus (trainseries) to keep

    Returns
    -------
    traintweet : pandas.DataFrame
        return the pandas dataframe associated with the trainseries where
        the top k words serve as the column names and the values in the 
        frame are the binary representation associated with each tweet.
    testtweet : pandas.DataFrame
        return the pandas dataframe associated with the testseries where
        the top k words serve as the column names and the values in the 
        frame are the binary representation associated with each tweet.
    """
    return None, None


def extract_tfidf(trainseries, testseries, k):
    """
    Extract relevant TF-IDF numeric word features from the tweet features
    
    Parameters
    ----------
    twseries : pandas.Series
        the original pandas dataframe
    k : int
        the number of top k words (in terms of term frequency) to keep

    Returns
    -------
    df : pandas.DataFrame
        return the pandas dataframe with the words as the column names
        and numeric TF-IDF representation in all the entries.
    """
    return None, None


def _reorder_columns(df, target='3_DAY_RETURN'):
    """
    Re-order the columns so the target is the last one
    """
    df = df[[col for col in df.columns if col != target] + [target]]
    return df


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("intrain", 
                        help="filename of training data")
    parser.add_argument("intest", 
                        help="filename of test data")
    parser.add_argument("k", type=int, 
                        help="number of top k words")
    parser.add_argument("otrain",
                        help="filename of the new training data")
    parser.add_argument("otest",
                        help="filename of the new test data")
    parser.add_argument('-b', 
                        '--binary', 
                        action='store_true',
                        help="use binary representation")
    args = parser.parse_args()
    train_df = pd.read_csv(args.intrain)
    test_df = pd.read_csv(args.intest)

    # extract dates for both train/test
    proc_train = extract_date(train_df)
    proc_test = extract_date(test_df)
    # extract company features for both/test
    proc_train = extract_company(proc_train)
    proc_test = extract_company(proc_test)

    # preprocess the data
    proc_train, proc_test = preprocess_data(proc_train, proc_test)

    # deal with the tweet representations
    if args.binary:
        print("Binary Representation (k=" + str(args.k) + ")")
        twtrain, twtest = extract_binary(proc_train["TWTOKEN"], 
                                         proc_test["TWTOKEN"],
                                         args.k)
    else:
        print("TF-IDF Representation (k=" + str(args.k) + ")")
        twtrain, twtest = extract_tfidf(proc_train["TWTOKEN"], 
                                        proc_test["TWTOKEN"],
                                        args.k)
    # append to the dataframe
    proc_train = pd.concat([proc_train, twtrain], axis=1)
    proc_test = pd.concat([proc_test, twtest], axis=1)
    # drop the text data
    proc_train.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
    proc_test.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
    # do some book-keeping here to make sure columns are the same
    ptrain, ptest = proc_train.align(proc_test,
                                     join='outer',
                                     axis=1, fill_value=0)
    # move the target to the last column
    ptrain = _reorder_columns(ptrain)
    ptest = _reorder_columns(ptest)

    print("Train Dataframe ---- ")
    print(ptrain)
    print("Test Dataframe Shape:", ptest.shape)

    # save to file, at this point should be numeric!
    ptrain.to_csv(args.otrain, index=False)
    ptest.to_csv(args.otest, index=False)


if __name__ == "__main__":
    main()