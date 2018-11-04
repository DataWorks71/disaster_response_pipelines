import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and join/merge datafiles to single dataset.

    Args:
    messages_filepath: str. The relative filepath to message file.  
    categories_filepath: str. The relative filepath to categories file.

    Returns:
    df: pandas.DataFrame. Merged dataset of messages and categories files.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    print('Number of records read...\n    MESSAGES: {0}\n    CATEGORIES: {1}'
          .format(len(messages), len(categories)))

    df = messages.merge(categories, how='inner', on='id')
    print('Number of merged records loaded...\n    MERGED(messages & categories): {0}'
          .format(len(df)))
       
    return df        


def clean_data(df):
    """Cleans dataset.
    -   Split `categories` into separate category columns.
    -   De-duplicate observations by deletion.
    -   Delete observations considered non-usable.

    Args:
    df: pandas.DataFrame. Dataset of messages and categories.

    Returns:
    df: pandas.DataFrame. Dataset of messages and categories.
    """
    # ### Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character 
    #   so that each value becomes a separate column. 
    #   https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', n=-1, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use selected row to extract a list of new column names for categories.
    # Takes everything up to the second to last character of each string.
    category_colnames = [str(x).split(sep='-')[0] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only 
    #   the last character of each string (the 1 or 0). 
    #   For example, `related-0` becomes `0`, `related-1` becomes `1`. 
    #   Convert the string to a numeric value.
    #   https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str).

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        
    # ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories, how='inner')
    
    print('`categories` was split into separate category columns')
    
    # ### 6. Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.

    # check number of duplicates
    n_duplicates = df.duplicated().sum()

    # calculate total number of rows
    n_rows_total = len(df)
    
    # drop duplicates
    df=df[~df.duplicated()]
    
    # calculate total number of rows
    n_rows = len(df)

    # check number of duplicates
    if n_duplicates > 0:
        if n_duplicates + n_rows == n_rows_total:
            print('Delete duplicate rows: CONFIRMED')
        else:
            print('Delete duplicate rows: NOT CONFIRMED')
    else:
        print('No duplicate rows detected')
 
    print('Number of duplicate rows deleted...\n    DUPLICATED ROWS: {0} of {1} deleted (rows left: {2})'
          .format(n_duplicates, n_rows_total, n_rows))
    
    ### Clean message, delete == `#NAME?`

    # calculate total number of rows
    n_rows_total = len(df)
    
    # delete messages without meaning 
    df = df[df.message!='#NAME?']
    
    # calculate number of rows left
    n_rows = len(df)
    
    print('Number of rows deleted...\n    `message`==`#NAME?`: {0} of {1} deleted (rows left: {2})'
          .format(n_rows_total - n_rows, n_rows_total, n_rows))

    
    ### Drop non-labeled observations, 
    # i.e 'related' == 2 is non-intepreted or non-translated messages 
    # without any positive category labels
    
    # calculate total number of rows
    n_rows_total = len(df)
        
    # Drop 'related' not in [0, 1].
    df = df[df.related.isin([0, 1])]
    
    # calculate number of rows left
    n_rows = len(df)
    
    print('Number of rows deleted...\n    `related` not in [0,1]: {0} of {1} deleted (rows left: {2})'
          .format(n_rows_total - n_rows, n_rows_total, n_rows))
 

    ### Drop any duplicated `id`s (keep none of the duplicates)
    # check number of duplicated 
    n_duplicates = df.duplicated('id', keep=False).sum()

    # calculate total number of rows
    n_rows = len(df)
    
    # drop duplicate `id`s - keep none
    df=df[~df.duplicated('id', keep=False)]
    
 
    print('Number of rows deleted...\n    DUPLICATED `id`s: {0} of {1} deleted (rows left: {2})'
          .format(n_duplicates, n_rows, n_rows-n_duplicates))
  

    return df


def save_data(df, database_filename):
    # ### 7. Save the clean dataset into an sqlite database. See:
    # - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html 
    # - SQLAlchemy library. 
    
    engine = create_engine('sqlite:///' + database_filename)
    my_table_name = 'ModelTrainData'
    
    df.to_sql(my_table_name, engine)
        

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