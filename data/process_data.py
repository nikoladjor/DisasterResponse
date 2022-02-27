import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Loads the disaster messages and categories into dataset.
    The values in the resulting pandas.Dataframe are converted to integer values
    relating message to the categories in separate columns.

    Args:
        messages_filepath: path to the messages .csv file.
        categories_filepath: path to the categories .csv file.

    Returns:
        pd.Dataframe

    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages_df.merge(categories_df, on='id')

    # Process the dataset
    # Categories column is one string --> this needs to be replaced

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x.split('-')[0]).values)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # From the columns, extract the category values
    # example: related-0, will have value 0 in column named "related"
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int32)

    # drop the original categories column from `df`
    df.drop(labels=["categories"],axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],sort=False, axis=1)

    return df


def clean_data(df) -> pd.DataFrame:
    """Clean up the data frame

    Args:
        df (pd.DataFrame): Input data frame.

    Returns:
        pd.DataFrame: Data frame with duplicated messages removed.
    """
    # drop duplicates
    df = df.drop_duplicates(subset=['message'])

    # Assert binary values in classification
    mvals = df.iloc[:,4:]
    
    to_drop = (mvals > 1).any(axis=1)
    df_clean = df.drop(to_drop[to_drop].index)

    assert(max(df_clean['id'].value_counts() == 1))
    return df_clean


def save_data(df, database_filename) -> None:
    """Save data frame to a database.

    Args:
        df (pd.DataFrame): Data frame to be saved.
        database_filename (str): Path to database to be saved.
    
    Returns: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename.split('.db')[0], engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()