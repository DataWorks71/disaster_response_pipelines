import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # pass
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    for df in [messages, categories]:
        df.set_index('id', inplace=True)
    
    df = messages.join(categories, how='outer')
    
    return df

def clean_data(df):
    # pass
    # ### 3. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character 
    #   so that each value becomes a separate column. 
    #   https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', n=-1, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
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
    df = df.join(categories, how='outer')


    # ### 6. Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.

    # check number of duplicates
    n_duplicates = df.duplicated().sum()

    # drop duplicates
    df=df[~df.duplicated()]

    # check number of duplicates
    if df.duplicated().sum() > 0:
        # Someting to console or log file
        pass
    return df

def save_data(df, database_filename):
    # ### 7. Save the clean dataset into an sqlite database. See:
    # - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html 
    # - SQLAlchemy library. 
    # pass  
    
    # TODO: Check for engine existence
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=True)


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