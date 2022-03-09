
from pyspark.sql import SparkSession
spark = SparkSession.builder\
        .appName("classification_first")\
        .getOrCreate()
spark
    

from pyspark.sql.functions import *
from pyspark.sql.types import (IntegerType,
                               FloatType,
                               StringType)
from pyspark.ml.feature import (StringIndexer,
                                VectorAssembler,
                                MinMaxScaler,
                                StandardScaler)

from pyspark.ml.classification import (RandomForestClassifier,
                                       GBTClassifier)

from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

from pyspark.ml.tuning import (CrossValidator,
                               ParamGridBuilder)

def convert_cat_type(
    df,
    col_name:str):

    indexed = StringIndexer(
        inputCol=col_name,
        outputCol=f'{col_name}_num'
    )
    df_in = indexed.fit(df).transform(df)

    return df_in



def cont_cat_split(df, independent_vars):

    cont_columns = []
    cat_columns = []

    for i in independent_vars:
        if str(df.schema[i].dataType) == 'StringType':

            new_column_name = f'{i}_num'
            df = convert_cat_type(
                df,
                col_name=i)

            cat_columns.append(new_column_name)
        else:
            cont_columns.append(i)

    return df, cont_columns, cat_columns


def percentile_from_continuous_var(
    df,
    continuous_columns):

    d = { }

    for i in continuous_columns:
        d[i] = df.approxQuantile(i, [0.01, 0.99], 0.01)

    return d

def treat_outliers(
    indexed,
    numeric_inputs,
    d):

    for col in numeric_inputs:
        skew = indexed.agg(
            skewness(
                indexed[col])
        ).collect()
        skew = skew[0][0]
        if skew > 1:
            indexed = indexed.withColumn(
                col,
                log(when(df[col] < d[col][0],d[col][0])\
                    .when(indexed[col] >d[col][1], d[col][1])\
                    .otherwise(indexed[col]) +1).alias(col)
            )

            print(f'{col} has been treated for positive right skewness')
        elif skew < -1:
            indexed = indexed.withColumn(
                col,
                exp(
                    when(df[col] < d[col][0], d[col][0])\
                   .when(df[col] > d[col][1], d[col][1])\
                   .otherwise(df[col])
                   ).alias(col)
                   )
            print(f'{col} has been treated for negative (left) skewness')

    return indexed


def preprocess_csv(
    csv_file_name:str,
    dependent_var:str,
    treat_outlier=False,
    data_max = 1000,
    data_min = 0,
    train_split=0.7,
    test_split=0.3):

    df = spark.read.csv(
        csv_file_name,
        header=True,
        inferSchema=True)
    
    print('printing dataframe\n =====')
    print(df.limit(2).toPandas())

    print(f'number of rows {df.count()}')

    print(f'removing nan from columns')
    df = df.na.drop()
    print(f'number of rows {df.count()}')

    print(f' printing dataframe schema \n')
    print(df.printSchema())

    print(
        'This is a classification problem, showing distribution of dependent variable\n')
    
    dependent_var_distribution = df.groupBy(dependent_var).count().toPandas()

    print(f' === printing distribution of output \n ====')

    print(dependent_var_distribution)

    dependent_var_distribution.plot.barh()

    independent_vars = df.columns
    independent_vars.remove(dependent_var)

    print(f'\n == independent variables are {independent_vars} == \n')


    print(f' extracting continuous and categorical variables \n')

    print(f' ======  ')

    df, cont_vars, cat_vars = cont_cat_split(
        df,independent_vars)

    df_indexed = df

    if str(df_indexed.schema[dependent_var].dataType) == 'StringType':
        indexer = StringIndexer(
            inputCol=dependent_var,
            outputCol='label')
        df_indexed = indexer.fit(df_indexed).transform(df_indexed)
    else:
        df_indexed = df_indexed.withColumnRenamed(
            dependent_var,
            'label'
        )

   
    print(df_indexed.columns)

    print(f' continuous variables are ===  {cont_vars} === \n categorical variables == {cat_vars}')

    print(f' percentile of continuous columns extracting in a dictionary')

    dict_ = percentile_from_continuous_var(
    df_indexed,
        cont_vars)

    if treat_outlier:

        print(' Treating outliers === \n')

        df_indexed = treat_outliers(
            df_indexed,
            cont_vars,
                dict_)

    else: print(f'== outliers is not treated === ')

    features_list = cont_vars + cat_vars

    print(f' =====so now columns are \n ===========')

    print(df_indexed.columns)

    print(f'\n')
    print(f' ===================== features list ============\n')
    print(f'{features_list}')

    #df_indexed = df_indexed.withColumnRenamed(
        #dependent_var,'label')

    assembler = VectorAssembler(
        inputCols=features_list,
        outputCol='features'
    )
    df_new = assembler.transform(
            df_indexed).select('features', 'label')


    print(f' scaling features to Minmax\n minimum is = {data_min} and maximum is  {data_max}')
    scaler = MinMaxScaler(
        inputCol='features',
        outputCol='scaledFeatures',
        min=data_min,
        max=data_max
        )

    scaled_data = scaler.fit(df_new).transform(df_new)

    final_data = scaled_data.select(
        'label','scaledFeatures'
        )

    print(final_data.show(5))

    # Rename to default value
    final_data = final_data.withColumnRenamed(
        'scaledFeatures',
        'features'
        )

    print(f' printing again the reamed version of data\n')
    print(final_data.show(5))

    print('splitting the data to test and training set')

    df_train, df_test = final_data.randomSplit(
        [train_split, test_split]
    )
    return df_train, df_test, independent_vars

