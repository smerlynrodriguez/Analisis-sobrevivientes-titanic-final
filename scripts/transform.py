import numpy as np


def ExtractDatasetInfo(dataset):
    
    columns = dataset.columns
    shape = dataset.shape
    info = dataset.info()
    describe = dataset.describe()
    describe_categories = 	dataset.describe(include=['0'])
    
    return columns, shape, info, describe, describe_categories


def EdaAnalyze(dataset):
    
    survivors = dataset.groupby(['Survived']).count()['PassengerId']
    survivors_sex = dataset.groupby(['Survived','Sex']).count()['PassengerId']
    survivors_per_sex_graph = (survivors_sex.unstack(level=0).plot.bar())
    
    return survivors, survivors_sex, survivors_per_sex_graph


def DataProcessing(dataset):
    
    table = dataset[['Survived', 'Sex', 'Age', 'Pclass']]
    table_info = table[['Survived', 'Sex', 'Age', 'Pclass']].info()
    age_null_values = (table[table['Age'].isna()] .groupby(['Sex', 'Pclass']) .count()['PassengerId'].unstack(level=0))
    siblings_per_class = (table[table['Age'].isna()] .groupby(['SibSp', 'Parch']) .count()['PassengerId'].unstack(level=0))
    age_median = table['Age'].median()
    table['Age'] = table['Age'].fillna(28.0)
    table['Sex'] = table['Sex'].map({'female': 1, 'male': 0}).astype(int)
    table['FlagSolo'] = np.where((table['SibSp'] == 0) & (table['Parch'] == 0), 1, 0)
    fixed_table = table[['Survived', 'Sex', 'Age', 'Pclass', 'FlagSolo']]
    survivors_if_siblings = dataset.groupby(['Survived','FlagSolo']).count()['PassengerId']
    survivors_if_siblings_graph = (survivors_if_siblings.unstack(level=0).plot.bar())
    
    return table, table_info, age_null_values, siblings_per_class, age_median, fixed_table, survivors_if_siblings, survivors_if_siblings_graph

