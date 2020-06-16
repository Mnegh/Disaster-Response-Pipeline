# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # dropping duplicates
    messages.drop_duplicates(subset=['id'],inplace=True)
    categories.drop_duplicates(subset=['id'],inplace=True)
    
    # merging data based on id
    df = pd.merge(messages,categories, how='inner',left_on=['id'],right_on=['id'])
    
    return df, categories
 
def clean_data(df, categories):
    # creates a dataframe of the 36 individual categories columns
    categories_split = pd.DataFrame(categories['categories'].str.split(';',expand=True))
    # select the first row of the categories dataframe
    row =  categories_split.iloc[0]
    # take the label names from the first row
    category_colnames = row.apply(lambda x: x.partition('-')[0])
    # rename the categories column
    categories_split.columns = category_colnames
    
    # convert the label values to binary
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(lambda x: x.partition('-')[-1])

        # convert column from string to numeric
        categories_split[column] = categories_split[column].apply(lambda x: int(x))
    
    # drop the original categories column from `df`
    df.drop(columns = ['categories'],inplace=True)
    
    # resetting indices to avoid NaN bug in concatenation
    df.reset_index(drop=True, inplace=True)
    categories_split.reset_index(drop=True, inplace=True)
    
    # concatenate the original df with the new cateogries dataframe
    df = pd.concat([df,categories_split],axis=1)
    
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Data', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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