import argparse
import pandas as pd

def create_split(df):
    """
    Create the train-test split. The method should be 
    randomized so each call will likely yield different 
    results.
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    train_df : pandas.DataFrame
        return the training dataset as a pandas dataframe
    test_df : pandas.DataFrame
        return the test dataset as a pandas dataframe.
    """
    return None, None


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="filename of training data")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    train_df, test_df = create_split(df)
    # solution here
    nonext = os.path.splitext(args.input)[0]
    print("Training DF Shape:", train_df.shape)
    train_df.to_csv(nonext+"_train.csv", index=False)
    print("Test DF Shape:", test_df.shape)
    test_df.to_csv(nonext+"_test.csv", index=False)


if __name__ == "__main__":
    main()
